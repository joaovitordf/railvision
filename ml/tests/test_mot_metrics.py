"""Unit tests for CLEAR MOT (MOTA, IDSW, FM) and global IDF1 calculation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_open_top_tracking_mot import evaluate_mot_sequence  # noqa: E402


def test_perfect_tracking_returns_ones_and_zero_errors() -> None:
    gt = {
        1: [(1, 100.0, 100.0, 200.0, 100.0), (2, 400.0, 100.0, 200.0, 100.0)],
        2: [(1, 110.0, 100.0, 200.0, 100.0), (2, 410.0, 100.0, 200.0, 100.0)],
        3: [(1, 120.0, 100.0, 200.0, 100.0), (2, 420.0, 100.0, 200.0, 100.0)],
    }
    pred = {
        1: [(10, 100.0, 100.0, 200.0, 100.0), (20, 400.0, 100.0, 200.0, 100.0)],
        2: [(10, 110.0, 100.0, 200.0, 100.0), (20, 410.0, 100.0, 200.0, 100.0)],
        3: [(10, 120.0, 100.0, 200.0, 100.0), (20, 420.0, 100.0, 200.0, 100.0)],
    }
    m = evaluate_mot_sequence(gt, pred, [1, 2, 3], iou_thresh=0.50)
    assert m["mota"] == 1.0
    assert m["idf1"] == 1.0
    assert m["idsw"] == 0
    assert m["fm"] == 0
    assert m["tp"] == 6
    assert m["fp"] == 0
    assert m["fn"] == 0


def test_single_id_switch_penalizes_mota_and_idf1() -> None:
    gt = {
        1: [(1, 100.0, 100.0, 200.0, 100.0)],
        2: [(1, 110.0, 100.0, 200.0, 100.0)],
        3: [(1, 120.0, 100.0, 200.0, 100.0)],
        4: [(1, 130.0, 100.0, 200.0, 100.0)],
    }
    pred = {
        1: [(7, 100.0, 100.0, 200.0, 100.0)],
        2: [(7, 110.0, 100.0, 200.0, 100.0)],
        3: [(9, 120.0, 100.0, 200.0, 100.0)],
        4: [(9, 130.0, 100.0, 200.0, 100.0)],
    }
    m = evaluate_mot_sequence(gt, pred, [1, 2, 3, 4], iou_thresh=0.50)
    assert m["tp"] == 4
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["idsw"] == 1
    assert m["fm"] == 0
    assert m["mota"] == 0.75
    assert m["idf1"] == 0.5


def test_track_gap_and_reentry_counts_fragmentation_and_fn() -> None:
    gt = {
        1: [(1, 100.0, 100.0, 200.0, 100.0)],
        2: [(1, 110.0, 100.0, 200.0, 100.0)],
        3: [(1, 120.0, 100.0, 200.0, 100.0)],
    }
    pred = {
        1: [(5, 100.0, 100.0, 200.0, 100.0)],
        2: [],
        3: [(5, 120.0, 100.0, 200.0, 100.0)],
    }
    m = evaluate_mot_sequence(gt, pred, [1, 2, 3], iou_thresh=0.50)
    assert m["tp"] == 2
    assert m["fn"] == 1
    assert m["fp"] == 0
    assert m["idsw"] == 0
    assert m["fm"] == 1


def test_empty_predictions_and_pure_false_positives() -> None:
    gt = {
        1: [(1, 100.0, 100.0, 200.0, 100.0)],
        2: [(1, 110.0, 100.0, 200.0, 100.0)],
    }
    m_empty = evaluate_mot_sequence(gt, {1: [], 2: []}, [1, 2], iou_thresh=0.50)
    assert m_empty["mota"] == 0.0
    assert m_empty["idf1"] == 0.0
    assert m_empty["fn"] == 2

    m_fp = evaluate_mot_sequence(
        gt,
        {
            1: [(3, 900.0, 800.0, 100.0, 50.0), (4, 1200.0, 800.0, 100.0, 50.0)],
            2: [(3, 905.0, 800.0, 100.0, 50.0)],
        },
        [1, 2],
        iou_thresh=0.50,
    )
    assert m_fp["tp"] == 0
    assert m_fp["fp"] == 3
    assert m_fp["fn"] == 2
    assert m_fp["mota"] == -1.5
    assert m_fp["idf1"] == 0.0
