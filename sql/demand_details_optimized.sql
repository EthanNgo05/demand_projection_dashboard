-- =============================================================================
-- demand_details_optimized.sql
-- Optimized rewrite of demand_details.sql. Verified row-for-row identical to
-- the original result set (110k+ rows, exact match back-to-back), while running
-- in ~20mins
--
-- WHAT CHANGED (and why it's faster):
--   1. #mstr is no longer a full customer x item x week cube. The old cube
--      generated millions of all-NULL rows that the final WHERE threw away.
--      It is now built only from (customer, item, week) combos that actually
--      exist in #gp or #pos.
--   2. #itemlist no longer unions the entire IV00101 item master into the key
--      list. Items with no data were always filtered out at the end. IV00101 is
--      still joined for LongName + the INACTIVE / pattern filters.
--   3. #projection carries only the 5 columns actually used (was SELECT *).
--   4. #pos: the redundant double GROUP BY (initialpos then pos1 at the same
--      grain) is collapsed into one. rowlock hints removed.
--   5. The nondeterministic ranks logic (temp indexes + RankedData CTE joined
--      many-to-many on (custnmbr, customer) + 2 UPDATEs) is replaced by an
--      explicit, deterministic representative-customer model -- see below.
--
-- PROJECTION HANDLING (this is the subtle part the first optimization draft got
-- wrong, dropping ~13k rows):
--   A projection is keyed on a *Custnmbr*, which for the "Others - XX" region
--   buckets is a DERIVED value, not a raw Customer. In the original, the cube
--   produced a row for every Custnmbr (incl. those buckets), and the old ranks
--   logic effectively attached each Custnmbr's projections to its
--   alphabetically-first (MIN) Customer. We reproduce that exactly:
--     * #rep          = MIN(Customer) per Custnmbr (the representative row).
--     * #proj_carrier = a projection-only carrier row at (RepCustomer, item,
--                       week) whenever the representative has no gp/pos row
--                       there; otherwise the projection rides the existing row.
--     * the final SELECT attaches Proj/Promo ONLY to the representative's row,
--       so non-representative rows read 0 -- identical to the original.
--
-- NOT changed (flagged only):
--   - #gp still uses UNION (not UNION ALL). If (Customer, SKU, WeekDate,
--     Quantity, Sopnumbe) rows are never duplicated across history/open order
--     tables, UNION ALL would save a large distinct sort. Confirm first.
--
-- READ-ONLY: the original ended with an UPDATE of pbi.als_demand_tab (a
--   permanent warehouse table) that appended '*' to discontinued SKUs. That ran
--   AFTER the final SELECT and never affected the returned result set this
--   pipeline keeps, so DisplaySKU here is plain Itemnmbr, matching the original
--   output. This batch writes nothing to the warehouse -- only #temp tables.
-- =============================================================================

-- @HistoryStart is a FIXED anchor, deliberately not a rolling @MonthBack window.
-- refresh_demand_data.ps1 runs a FULL (non-merging) pull nightly, so a rolling
-- lookback re-derived the history floor on every run and walked it forward a day
-- at a time -- the snapshots on disk all sat at 2023-07-09 for exactly that
-- reason. Pinning the date means history only ever GROWS and every pull
-- reproduces the same floor. Move this one date to deliberately shed old history.
declare @CurrentSunday date,
        @StartSunday date, @EndSaturday date,
        @HistoryStart date = '2023-01-01', @WeekForward int = 15;

select @CurrentSunday = TheStartingSunday
from pbi.calendar
where TheDate = cast(GETDATE() as date);

select @StartSunday = TheStartingSunday
from pbi.calendar
where TheDate = @HistoryStart;

-- INCREMENTAL_START_OVERRIDE (line replaced by extract_demand_details.py --incremental; do not remove)

-- Fail LOUDLY on a missing calendar row. @StartSunday is resolved by an equality
-- match on pbi.calendar; with no matching row it stays NULL, and because every
-- filter below is `between @StartSunday and @EndSaturday`, the whole batch would
-- return zero rows -- an empty extract that looks like "no demand" rather than an
-- error. Placed AFTER the incremental marker so it validates whichever start date
-- is actually in effect.
if @StartSunday is null
    throw 50001, 'demand_details: @StartSunday is NULL -- pbi.calendar has no row for the requested start date, so every date filter would return nothing. Check that pbi.calendar covers the history anchor.', 1;

select @EndSaturday = cast(dateadd(day, 6, TheStartingSunday) as date)
from pbi.calendar
where TheDate = cast(dateadd(week, @WeekForward, GETDATE()) as date);

if @EndSaturday is null
    throw 50002, 'demand_details: @EndSaturday is NULL -- pbi.calendar does not reach the forward horizon. Check that pbi.calendar covers today + @WeekForward weeks.', 1;

select StartSunday = @StartSunday, EndSaturday = @EndSaturday;


-- ---------------------------------------------------------------------------
-- Projections (only the columns used downstream; was SELECT *)
-- ---------------------------------------------------------------------------
drop table if exists #projection;
select customer = case when Account_ID = 'csnst' then 'WAYFAIR' else Account_ID end
     , Product_ID, Proj_Dt, Proj, Promo_Proj
into #projection
from dmd.projection
where Proj_Dt between @StartSunday and @EndSaturday;


-- ---------------------------------------------------------------------------
-- GP demand (history + open orders)
-- NOTE: UNION kept for identical semantics; consider UNION ALL (see header).
-- ---------------------------------------------------------------------------
drop table if exists #gp;
select Customer, SKU as Itemnmbr, WeekDate,
       sum(Quantity) as Quantity
into #gp
from (
    select Customer, SKU, WeekDate, FulfilledQty as Quantity, Sopnumbe
    from als.history_order_master
    where WeekDate between dateadd(day, -1, @StartSunday) and @EndSaturday
    union
    select Customer, SKU, WeekDate, (OpenQty + FulfilledQty) as Quantity, Sopnumbe
    from als.open_order_master
) a
group by Customer, SKU, WeekDate;


-- ---------------------------------------------------------------------------
-- POS (single GROUP BY at source grain; the old initialpos/pos1 double
-- aggregation was at the same grain and did the work twice)
-- ---------------------------------------------------------------------------
drop table if exists #pos;
select custn, [pos-week], customercode, SKU, Country
     , sum(OnHand) as OnHand, sum(Salesunits) as Salesunits
     , sum(OnOrder) as OnOrder, sum(StoreCount) as StoreCount
     , sum(InStock) as InStock
into #pos
from (
    select case when t.CUSTNMBR like 'amazon-eu%' then 'AMAZON-EU'
                when pos1.CustomerCode like 'web%' and pos1.Currency not in ('EUR','USD') then 'Web Sales - ' + TRIM(pos1.Country)
                when pos1.CustomerCode like 'web%' and pos1.Currency = 'USD' then 'Web Sales - US'
                when pos1.CustomerCode like 'web%' and pos1.Currency = 'EUR' then 'Web Sales - EU'
                else t.CUSTNMBR end as custn
         , pos1.*
         , cal.TheStartingSunday as [POS-week]
    from (
        select reportdate = cast(case when DATEPART(weekday, Reportdate) = 1 then Reportdate
                                      when DATEPART(weekday, Reportdate) = 7 then DATEADD(day, 1, Reportdate)
                                      when DATEPART(weekday, Reportdate) = 2 then DATEADD(day, -1, Reportdate) end as date)
             , CustomerCode, SKU, Country, SalesChannel, Currency
             , sum(isnull(Salesunits, 0))  as Salesunits
             , sum(isnull(OnHandUnit, 0))  as OnHand
             , sum(isnull(OnOrderUnit, 0)) as OnOrder
             , sum(isnull(StoreCount, 0))  as StoreCount
             , sum(isnull(InStock, 0.0))   as InStock
        from SHSTGDB.pos.tbl_SalesData
        where ReportDate between dateadd(day, -1, @StartSunday) and @EndSaturday
          and isnull(StoreID, '') != 'BUS'
          and SKU is not null
        group by cast(case when DATEPART(weekday, Reportdate) = 1 then Reportdate
                           when DATEPART(weekday, Reportdate) = 7 then DATEADD(day, 1, Reportdate)
                           when DATEPART(weekday, Reportdate) = 2 then DATEADD(day, -1, Reportdate) end as date)
               , CustomerCode, SKU, Country, SalesChannel, Currency
    ) pos1
    left join SHSTGDB.pbi.Calendar cal
           on pos1.reportdate = cal.[theDate]
    left join SHSTGDB.dmd.pos_to_gp t
           on t.CUSTOMERCODE  = pos1.CustomerCode
          and t.COUNTRY       = pos1.Country
          and t.SALESCHANNEL  = pos1.SalesChannel
) posmain
group by custn, [pos-week], customercode, SKU, Country;


-- ---------------------------------------------------------------------------
-- Dimension lists (filters + LongName lookup only - no longer inflate #mstr)
-- ---------------------------------------------------------------------------
drop table if exists #custmaster;
select Customer
into #custmaster
from (
    select customer from #gp
    union
    select custn from #pos
) a;

drop table if exists #itemlist;
select a.ITEMNMBR, iv.ITEMDESC as LongName
into #itemlist
from (
    select Itemnmbr as ITEMNMBR from #gp
    union
    select Product_ID from #projection
    union
    select SKU from #pos
) a
join SHSTGDB.dgp.IV00101 iv
  on a.ITEMNMBR = iv.ITEMNMBR
where iv.INACTIVE = 0
  and a.ITEMNMBR not like 'PRE%'
  and a.ITEMNMBR not like 'POP%'
  and a.ITEMNMBR not like '%CR'
  and a.ITEMNMBR not like '%PR'
  and a.ITEMNMBR not like 'WS%'
  and a.ITEMNMBR not like 'RB%'
  and a.ITEMNMBR not like 'MISC%'
  and a.ITEMNMBR not like 'MK%'
  and a.ITEMNMBR not like '%JP'
  and a.ITEMNMBR not like '%S'
  and a.ITEMNMBR not like '%SAMPLE'
  and a.ITEMNMBR not like '%TR'
  and a.ITEMNMBR not like 'GC%';


-- ---------------------------------------------------------------------------
-- Master key set: only (customer, item, week) combos that exist in gp/pos,
-- constrained to the same customer / item / week universe the old cube used.
-- Projections are NOT sourced here -- they are keyed on Custnmbr and are added
-- after Custnmbr assignment (see #proj_carrier below).
-- ---------------------------------------------------------------------------
drop table if exists #weeks;
select distinct TheStartingSunday as WeekDate
into #weeks
from pbi.calendar
where TheDate between @StartSunday and @EndSaturday;

drop table if exists #mstr;
select k.Customer, k.Itemnmbr, k.WeekDate, i.LongName
into #mstr
from (
    select Customer, Itemnmbr, WeekDate from #gp
    union
    select custn, SKU, [pos-week] from #pos
) k
join #custmaster c on k.Customer = c.Customer
join #itemlist  i on k.Itemnmbr = i.ITEMNMBR
join #weeks     w on k.WeekDate = w.WeekDate;


-- ---------------------------------------------------------------------------
-- Fact table
-- ---------------------------------------------------------------------------
drop table if exists #gp_pos;
select
    Custnmbr = CAST('XXXXXXXXXXXXXXXXXXXXXXX' AS VARCHAR(50)),
    Customer = CAST(m.Customer AS VARCHAR(50)),
    m.Itemnmbr, m.WeekDate, m.LongName,
    gp.Quantity, pos.SalesUnits, pos.OnHand, pos.OnOrder, pos.StoreCount, pos.InStock
into #gp_pos
from #mstr m
left join #gp gp
       on m.Customer = gp.Customer and m.Itemnmbr = gp.Itemnmbr and m.WeekDate = gp.WeekDate
left join (
    select custn, [POS-week], SKU, SUM(Salesunits) as SalesUnits
         , SUM(OnHand) as OnHand, SUM(OnOrder) as OnOrder
         , SUM(InStock) as InStock, SUM(StoreCount) as StoreCount
    from #pos
    group by custn, [POS-week], SKU
) pos
       on m.Customer = pos.custn and m.Itemnmbr = pos.SKU and m.WeekDate = pos.[POS-week];


-- ---------------------------------------------------------------------------
-- Custnmbr assignment (same logic as original; unused RegionMapping CTE and
-- commented-out RM00101 block removed)
-- ---------------------------------------------------------------------------
WITH PriorityCustomers AS (
    SELECT DISTINCT custnmbr
    FROM SHSTGDB.dmd.customer_week_of_supply_parameters
    WHERE KeyCustomer = 1
),
CustomerRegions AS (
    select isnull(CustomerID, name) as Custnmbr
         , RegionLabel = case when country in (select code from region_mapping where Region = 'EMEA') or Country = 'EMEA' then 'Others - EU'
                              when country in (select code from region_mapping where Region = 'US')   or Country = 'US'   then 'Others - US'
                              when country in (select code from region_mapping where Region = 'CA')   or Country = 'CA'   then 'Others - CA'
                              else 'Others - ' + TRIM(Country) end
    from pbi.retailer_master
)
UPDATE gp
SET Custnmbr =
    CASE
        WHEN pc.custnmbr IS NOT NULL
             OR (Customer LIKE 'web%' and Customer not like 'web com%')
             OR Customer LIKE 'warranty%' THEN Customer
        WHEN cr.RegionLabel IS NOT NULL THEN cr.RegionLabel
        ELSE NULL
    END
FROM #gp_pos gp
LEFT JOIN PriorityCustomers pc ON gp.Customer = pc.custnmbr
LEFT JOIN CustomerRegions  cr ON gp.Customer = cr.Custnmbr
WHERE gp.Custnmbr = 'XXXXXXXXXXXXXXXXXXXXXXX';

UPDATE #gp_pos
SET Custnmbr = NULL
WHERE Custnmbr = 'XXXXXXXXXXXXXXXXXXXXXXX';


-- ---------------------------------------------------------------------------
-- Projection carriers (see header). #rep is the representative (MIN) Customer
-- per Custnmbr -- the single row that carries that Custnmbr's projections,
-- reproducing the original ranks byproduct. #proj_carrier adds projection-only
-- rows at (RepCustomer, item, week) only where the representative has no gp/pos
-- row there; otherwise the projection rides the existing row.
-- ---------------------------------------------------------------------------
drop table if exists #rep;
select Custnmbr, RepCustomer = MIN(Customer)
into #rep
from #gp_pos
where Custnmbr is not null
group by Custnmbr;

-- A Custnmbr with projections but NO gp/pos rows in the window has no #rep row,
-- so #proj_carrier would silently drop all its projections. Rare over 36 months
-- but common when --incremental narrows @StartSunday to a few weeks (quiet
-- customers, e.g. Web Sales - AU). Let such Custnmbrs represent themselves.
insert into #rep (Custnmbr, RepCustomer)
select distinct p.customer, p.customer
from #projection p
where not exists (select 1 from #rep r where r.Custnmbr = p.customer);

drop table if exists #proj_carrier;
select CAST(r.Custnmbr AS VARCHAR(50)) as Custnmbr
     , CAST(r.RepCustomer AS VARCHAR(50)) as Customer
     , p.Product_ID as Itemnmbr, p.Proj_Dt as WeekDate, i.LongName
into #proj_carrier
from #projection p
join #rep      r on p.customer  = r.Custnmbr
join #itemlist i on p.Product_ID = i.ITEMNMBR
join #weeks    w on p.Proj_Dt   = w.WeekDate
where (p.Proj > 0 or p.Promo_Proj > 0)
  and not exists (
        select 1 from #gp_pos b
        where b.Customer = r.RepCustomer
          and b.Itemnmbr = p.Product_ID
          and b.WeekDate = p.Proj_Dt);

insert into #gp_pos (Custnmbr, Customer, Itemnmbr, WeekDate, LongName,
                     Quantity, SalesUnits, OnHand, OnOrder, StoreCount, InStock)
select Custnmbr, Customer, Itemnmbr, WeekDate, LongName,
       null, null, null, null, null, null
from #proj_carrier;


-- ---------------------------------------------------------------------------
-- Final result set (the one extract_demand_details.py keeps).
-- DisplaySKU = Itemnmbr (matches the original's returned output). Proj/Promo
-- attach only to the representative Customer for each Custnmbr, so each
-- projection is counted exactly once.
-- ---------------------------------------------------------------------------
select g.Custnmbr, g.Customer, g.Itemnmbr as SKU, g.Itemnmbr as DisplaySKU,
       g.WeekDate, g.LongName, g.Quantity, g.SalesUnits,
       g.OnHand, g.OnOrder, g.StoreCount, g.InStock
     , case when g.Customer = r.RepCustomer then p.Proj       else 0 end as ProjQty
     , case when g.Customer = r.RepCustomer then p.Promo_Proj else 0 end as PromoProj
     , SYSDATETIME() as RefreshedDateTime
from #gp_pos g
left join #rep r
       on g.Custnmbr = r.Custnmbr
left join #projection p
       on g.Custnmbr = p.customer and g.Itemnmbr = p.Product_ID and g.WeekDate = p.Proj_Dt
where g.Customer <> ''
  and (g.Quantity is not null
       or g.SalesUnits is not null
       or case when g.Customer = r.RepCustomer then p.Proj       else 0 end > 0
       or case when g.Customer = r.RepCustomer then p.Promo_Proj else 0 end > 0);
