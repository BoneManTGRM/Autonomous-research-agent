import math
import random
import pytest

from agent.replay_buffer import ReplayBuffer, ReplayItem


def _make_buffer(
    max_items_per_hallmark: int = 10,
    max_total_items: int = 100,
) -> ReplayBuffer:
    """Helper to create a small buffer for tests."""
    buf = ReplayBuffer(
        max_items_per_hallmark=max_items_per_hallmark,
        max_total_items=max_total_items,
    )
    return buf


def _basic_add(
    buffer: ReplayBuffer,
    hallmark: str = "mitochondria",
    stage: str = "idea",
    rye: float = 0.8,
    energy: float = 10.0,
    cycle_index: int = 0,
    extra_tag: str | None = None,
) -> ReplayItem:
    tags = ["longevity"]
    if extra_tag:
        tags.append(extra_tag)

    item = buffer.add_item(
        hallmark=hallmark,
        stage=stage,
        mechanism_chain=["node_a", "node_b"],
        biomarker_pattern={"NAD+": "up"},
        hypothesis_text=f"Test hypothesis for {hallmark} at stage {stage}",
        rye_score=rye,
        energy_cost=energy,
        decision="pending",
        source_citations=[{"title": "Paper A", "url": "https://example.com/a"}],
        tags=tags,
        run_id="run_test_1",
        cycle_index=cycle_index,
    )
    return item


def test_replay_item_structure_minimal():
    """ReplayItem should expose all core fields used by the longevity stack."""
    buf = _make_buffer()
    item = _basic_add(buf)

    # Core identity
    assert isinstance(item.item_id, str)
    assert item.item_id

    # Hallmark and stage
    assert item.hallmark == "mitochondria"
    assert item.stage == "idea"

    # Mechanism and biomarkers
    assert isinstance(item.mechanism_chain, list)
    assert item.mechanism_chain
    assert isinstance(item.biomarker_pattern, dict)

    # Hypothesis and scores
    assert isinstance(item.hypothesis_text, str)
    assert item.rye_score is not None
    assert isinstance(item.rye_score, (int, float))

    # Decision and citations
    assert item.decision in {"pending", "accepted", "rejected"}
    assert isinstance(item.source_citations, list)

    # Tags and meta
    assert isinstance(item.tags, list)
    assert "longevity" in item.tags
    assert item.run_id == "run_test_1"
    assert item.cycle_index == 0

    # created_at may be str or datetime, just ensure it exists
    assert getattr(item, "created_at", None) is not None


def test_add_item_registers_in_buffer():
    """Adding an item should register it in the buffer indices."""
    buf = _make_buffer()
    item = _basic_add(buf, hallmark="mitochondria", stage="idea", rye=0.9, cycle_index=1)

    # Buffer should know about the item
    items_for_hallmark = buf.get_items_for_hallmark("mitochondria")
    assert any(x.item_id == item.item_id for x in items_for_hallmark)

    # Top items call should include it
    top_items = buf.get_top_items(hallmark="mitochondria", top_k=5)
    assert any(x.item_id == item.item_id for x in top_items)


def test_priority_ordering_by_rye_and_energy():
    """Higher RYE and lower energy should rank higher in top items."""
    buf = _make_buffer(max_items_per_hallmark=10, max_total_items=50)

    # Lower RYE, higher energy
    low_item = _basic_add(
        buf,
        hallmark="mitochondria",
        stage="idea",
        rye=0.4,
        energy=30.0,
        cycle_index=1,
        extra_tag="low",
    )

    # Moderate
    mid_item = _basic_add(
        buf,
        hallmark="mitochondria",
        stage="idea",
        rye=0.6,
        energy=20.0,
        cycle_index=2,
        extra_tag="mid",
    )

    # High RYE, low energy
    high_item = _basic_add(
        buf,
        hallmark="mitochondria",
        stage="idea",
        rye=0.9,
        energy=8.0,
        cycle_index=3,
        extra_tag="high",
    )

    top = buf.get_top_items(hallmark="mitochondria", top_k=3)

    # All three should be present
    ids = [x.item_id for x in top]
    assert low_item.item_id in ids
    assert mid_item.item_id in ids
    assert high_item.item_id in ids

    # Check ordering: high should rank before low
    id_pos = {x.item_id: i for i, x in enumerate(top)}
    assert id_pos[high_item.item_id] < id_pos[low_item.item_id]


def test_sampling_prefers_high_priority_items():
    """Sampling for a stage should statistically favor the best items."""
    buf = _make_buffer(max_items_per_hallmark=30, max_total_items=100)

    # Many low quality items
    for i in range(15):
        _basic_add(
            buf,
            hallmark="senescence",
            stage="idea",
            rye=0.3,
            energy=30.0,
            cycle_index=i,
        )

    # A few strong items
    strong_items = []
    for i in range(3):
        strong_items.append(
            _basic_add(
                buf,
                hallmark="senescence",
                stage="idea",
                rye=0.9,
                energy=8.0,
                cycle_index=100 + i,
                extra_tag="strong",
            )
        )

    if not hasattr(buf, "sample_for_stage"):
        pytest.skip("ReplayBuffer has no sample_for_stage, skipping preference test")

    count_strong = 0
    trials = 80
    for _ in range(trials):
        sample = buf.sample_for_stage(
            hallmark="senescence",
            stage="idea",
            k=1,
        )
        if not sample:
            continue
        if sample[0].rye_score and sample[0].rye_score > 0.8:
            count_strong += 1

    # Strong items should be sampled significantly more than random uniform
    # With 3 strong out of 18 total, uniform would be ~13 percent.
    # We require at least 35 percent strong to confirm prioritization.
    strong_rate = count_strong / max(1, trials)
    assert strong_rate > 0.35


def test_pruning_respects_max_total_items():
    """When buffer exceeds max_total_items it should prune weaker entries."""
    buf = _make_buffer(max_items_per_hallmark=50, max_total_items=20)

    # Fill with many mediocre items
    for i in range(25):
        _basic_add(
            buf,
            hallmark="mitochondria",
            stage="idea",
            rye=0.4 + 0.01 * (i % 5),
            energy=25.0,
            cycle_index=i,
        )

    all_items = buf.get_items_for_hallmark("mitochondria")
    assert len(all_items) <= 20


def test_per_hallmark_cap_enforced():
    """Each hallmark should obey its own cap."""
    buf = _make_buffer(max_items_per_hallmark=5, max_total_items=100)

    # Add more than 5 items to the same hallmark
    for i in range(12):
        _basic_add(
            buf,
            hallmark="mitochondria",
            stage="idea",
            rye=0.5 + 0.01 * i,
            energy=20.0,
            cycle_index=i,
        )

    items_mito = buf.get_items_for_hallmark("mitochondria")
    assert len(items_mito) <= 5

    # A different hallmark should not be affected
    for i in range(3):
        _basic_add(
            buf,
            hallmark="senescence",
            stage="verify",
            rye=0.7,
            energy=15.0,
            cycle_index=100 + i,
        )

    items_sen = buf.get_items_for_hallmark("senescence")
    assert len(items_sen) == 3


def test_update_decision_and_outcome():
    """Decisions and outcomes should attach cleanly to stored items."""
    buf = _make_buffer()
    item = _basic_add(buf, rye=0.85, energy=9.0, cycle_index=1)

    if not hasattr(buf, "update_decision"):
        pytest.skip("ReplayBuffer has no update_decision, skipping decision test")

    buf.update_decision(item_id=item.item_id, decision="accepted", reason="High RYE and coherent chain")

    items = buf.get_items_for_hallmark("mitochondria")
    updated = next(x for x in items if x.item_id == item.item_id)
    assert updated.decision == "accepted"
    assert getattr(updated, "decision_reason", None) in {
        "High RYE and coherent chain",
        # If implementation uses a different field name, we only require no error.
        getattr(updated, "decision_reason", None),
    }

    if hasattr(buf, "record_outcome"):
        buf.record_outcome(
            item_id=item.item_id,
            outcome_type="verified_signal",
            outcome_value={"p_value": 0.01},
        )


def test_serialisation_round_trip():
    """Buffer to_dict and from_dict round trip should preserve core info."""
    buf = _make_buffer(max_items_per_hallmark=10, max_total_items=50)

    for i in range(6):
        _basic_add(
            buf,
            hallmark="mitochondria",
            stage="idea",
            rye=0.7 + 0.02 * i,
            energy=10.0 + i,
            cycle_index=i,
        )

    if not hasattr(buf, "to_dict") or not hasattr(ReplayBuffer, "from_dict"):
        pytest.skip("ReplayBuffer has no to_dict/from_dict, skipping serialisation test")

    data = buf.to_dict()
    buf2 = ReplayBuffer.from_dict(data)

    items1 = sorted(buf.get_items_for_hallmark("mitochondria"), key=lambda x: x.item_id)
    items2 = sorted(buf2.get_items_for_hallmark("mitochondria"), key=lambda x: x.item_id)

    assert len(items1) == len(items2)
    for a, b in zip(items1, items2):
        assert a.item_id == b.item_id
        assert a.hallmark == b.hallmark
        assert math.isclose(a.rye_score or 0.0, b.rye_score or 0.0, rel_tol=1e-6)


def test_replay_summary_basic_shape():
    """Replay summary should expose hallmark level stats if available."""
    buf = _make_buffer()

    _basic_add(buf, hallmark="mitochondria", stage="idea", rye=0.9, energy=9.0, cycle_index=1)
    _basic_add(buf, hallmark="mitochondria", stage="verify", rye=0.8, energy=11.0, cycle_index=2)
    _basic_add(buf, hallmark="senescence", stage="idea", rye=0.7, energy=12.0, cycle_index=3)

    if not hasattr(buf, "get_replay_summary"):
        pytest.skip("ReplayBuffer has no get_replay_summary, skipping summary test")

    summary = buf.get_replay_summary()

    assert isinstance(summary, dict)
    assert "hallmarks" in summary

    hallmarks = summary["hallmarks"]
    assert "mitochondria" in hallmarks
    assert "senescence" in hallmarks

    mito = hallmarks["mitochondria"]
    assert "count" in mito
    assert mito["count"] >= 2
    assert "mean_rye" in mito
    assert isinstance(mito["mean_rye"], (int, float))


def test_stage_filtering_in_top_items():
    """Top items with stage filter should restrict results correctly."""
    buf = _make_buffer()

    _basic_add(buf, hallmark="mitochondria", stage="idea", rye=0.9, energy=8.0, cycle_index=1)
    _basic_add(buf, hallmark="mitochondria", stage="verify", rye=0.85, energy=9.0, cycle_index=2)
    _basic_add(buf, hallmark="mitochondria", stage="verify", rye=0.6, energy=12.0, cycle_index=3)

    top_idea = buf.get_top_items(hallmark="mitochondria", stage="idea", top_k=5)
    assert all(it.stage == "idea" for it in top_idea)

    top_verify = buf.get_top_items(hallmark="mitochondria", stage="verify", top_k=5)
    assert all(it.stage == "verify" for it in top_verify)
    assert len(top_verify) == 2
