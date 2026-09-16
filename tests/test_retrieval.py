from src.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_prefers_items_ranked_highly_in_both_lists():
    dense = [{"chunk_id": "a", "text": "a"}, {"chunk_id": "b", "text": "b"}]
    sparse = [{"chunk_id": "b", "text": "b"}, {"chunk_id": "a", "text": "a"}]
    fused = reciprocal_rank_fusion([dense, sparse])
    # both appear once in each list at ranks {0,1}; scores should be equal
    # and both should be present, proving fusion merges rather than drops.
    assert {f["chunk_id"] for f in fused} == {"a", "b"}
    assert fused[0]["fused_score"] == fused[1]["fused_score"]


def test_rrf_ranks_item_present_in_both_lists_above_item_in_one():
    dense = [{"chunk_id": "a", "text": "a"}, {"chunk_id": "c", "text": "c"}]
    sparse = [{"chunk_id": "a", "text": "a"}, {"chunk_id": "b", "text": "b"}]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0]["chunk_id"] == "a"
