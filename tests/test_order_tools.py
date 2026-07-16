"""Tests for the stock-management agent tools (order place / undo / list),
including the consistent critical rule and the auto-computed order quantity."""

import pandas as pd
import pytest

import src.assistant_tools as at


@pytest.fixture(autouse=True)
def inject_inventory():
    # needs-order rule: current < reorder_point OR days_of_cover < CRIT_DAYS(2).
    #  P0001: 5<100            -> critical (below reorder)
    #  P0002: 8<50             -> critical
    #  P0003: 40<60            -> critical (below reorder, was "REORDER")
    #  P0004: 200>=50, 30d     -> OK, excluded
    #  P0005: 80>=60 but 1.0d  -> critical (days rule, even though above reorder)
    at._inv_cache = pd.DataFrame({
        "Store ID": ["S001", "S002", "S003", "S004", "S005"],
        "Product ID": ["P0001", "P0002", "P0003", "P0004", "P0005"],
        "abc_class": ["A", "A", "B", "C", "C"],
        "alert_status": ["CRITICAL", "CRITICAL", "REORDER", "OK", "OK"],
        "current_inventory": [5, 8, 40, 200, 80],
        "reorder_point": [100.0, 50.0, 60.0, 50.0, 60.0],
        "EOQ": [50.0, 40.0, 30.0, 20.0, 25.0],
        "days_of_cover": [0.5, 1.5, 4.0, 30.0, 1.0],
    })
    at.set_client_orders([])
    yield
    at._inv_cache = None
    at.set_client_orders([])


def test_recommended_order_formula():
    row = {"reorder_point": 100.0, "EOQ": 50.0, "current_inventory": 5}
    assert at.recommended_order(row) == 145           # max(0, 100+50-5)
    # Never negative when already well-stocked.
    assert at.recommended_order({"reorder_point": 10, "EOQ": 5, "current_inventory": 999}) == 0


def test_critical_rule_is_consistent_and_excludes_ok():
    d = at._critical_rows()
    ids = list(d["Product ID"])
    assert "P0004" not in ids                          # OK (above reorder, plenty of cover)
    assert "P0005" in ids                              # above reorder BUT < 2 days cover
    assert len(ids) == 4                               # count == the 4 needs-order SKUs


def test_siparis_ver_orders_by_urgency_with_quantities():
    out = at.siparis_ver(top_n=10)
    order = [(s["store_id"], s["product_id"], s["qty"]) for s in out["action"]["skus"]]
    # lowest days-of-cover first: P0001(.5), P0005(1.0), P0002(1.5), P0003(4.0)
    assert [o[1] for o in order] == ["P0001", "P0005", "P0002", "P0003"]
    assert out["count"] == 4
    # every quantity is the auto formula and strictly positive
    assert dict((o[1], o[2]) for o in order)["P0001"] == 145
    assert all(o[2] > 0 for o in order)


def test_siparis_ver_top_n_limits():
    out = at.siparis_ver(top_n=2)
    assert out["count"] == 2
    assert [s["product_id"] for s in out["action"]["skus"]] == ["P0001", "P0005"]


def test_siparis_ver_specific_sku_has_quantity():
    out = at.siparis_ver(store_id="S002", product_id="P0002")
    assert out["count"] == 1
    s = out["action"]["skus"][0]
    assert s["store_id"] == "S002" and s["product_id"] == "P0002"
    assert s["qty"] == 82                               # 50 + 40 - 8


def test_siparis_geri_al_single_and_all():
    one = at.siparis_geri_al(store_id="S001", product_id="P0001")
    assert one["action"] == {"op": "unmark", "skus": [{"store_id": "S001", "product_id": "P0001"}]}
    everything = at.siparis_geri_al(hepsi=True)
    assert everything["action"] == {"op": "unmark", "all": True}
    assert "error" in at.siparis_geri_al()              # nothing specified


def test_verilen_siparisleri_listele_reads_client_state():
    at.set_client_orders([{"store_id": "S001", "product_id": "P0001", "ts": 1, "qty": 145}])
    out = at.verilen_siparisleri_listele()
    assert out["count"] == 1
    assert out["orders"][0]["product_id"] == "P0001"
    assert out["orders"][0]["qty"] == 145
    assert out["action"]["op"] == "open_orders"
