"""One small runnable check: python test_pipeline.py."""
import tempfile
from pathlib import Path
from pipeline import clean_blocks, connect_db, crop_box, index_records, parse_json, search, pin_image, repair_groups


def check():
    blocks = clean_blocks({"parsing_res_list": [
        {"block_bbox": [10, 20, 90, 50], "block_content": "7. 已知函数"},
        {"block_bbox": [15, 55, 80, 95], "block_label": "image"},
        {"block_bbox": [10, 100, 90, 120], "block_content": "8. 下一题"},
    ]}, 100, 130)
    box = crop_box(blocks, [0, 1], 100, 130)
    assert box == [2, 12, 98, 103], box
    try:
        crop_box(blocks, [90], 100, 130)
        raise AssertionError("out-of-range block was accepted")
    except ValueError:
        pass
    assert parse_json('```json\n{"questions":[]}\n```')["questions"] == []
    repaired = repair_groups({"subject": "数学"}, blocks, {"page_kind": "question", "questions": [
        {"number": "7", "block_ids": [0], "status": "complete"},
        {"number": "8", "block_ids": [2], "status": "complete"}]})
    assert repaired["questions"][0]["block_ids"] == [0, 1], repaired
    assert repaired["questions"][-1]["status"] == "review", repaired
    truncated = [{"id": 0, "bbox": [0, 0, 100, 50], "label": "text", "text": "7. 给出下列四个结论：①函数为奇函数"}]
    assert repair_groups({"subject": "数学"}, truncated, {"questions": []})["questions"][0]["status"] == "cross_page"
    english = [
        {"id": 0, "bbox": [0, 0, 100, 20], "label": "text", "text": "continued reading passage"},
        {"id": 1, "bbox": [0, 20, 100, 40], "label": "table", "text": "schedule"},
        {"id": 2, "bbox": [0, 40, 100, 60], "label": "text", "text": "31. Question?"},
        {"id": 3, "bbox": [0, 60, 100, 70], "label": "paragraph_title", "text": "B"},
        {"id": 4, "bbox": [0, 70, 100, 90], "label": "image", "text": ""}]
    grouped = repair_groups({"subject": "英语"}, english, {"page_kind": "question", "questions": [
        {"number": "31", "block_ids": [2], "status": "complete"}]}, "previous page")
    assert grouped["questions"][0]["block_ids"] == [0, 1, 2], grouped
    assert grouped["questions"][0]["status"] == "cross_page", grouped
    assert grouped["questions"][1]["block_ids"] == [3, 4], grouped
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original = root / "q1.jpg"
        original.write_bytes(b"sample-image")
        stable = root / pin_image(original, "q1", root)
        renamed = root / "moved.jpg"
        original.rename(renamed)
        assert stable.read_bytes() == b"sample-image"
        renamed.write_bytes(b"edited-image")
        assert stable.read_bytes() == b"edited-image", "move must retain a shared file, not a copied snapshot"
        renamed.unlink()
        assert stable.read_bytes() == b"edited-image", "removing the export must not break the index"
        db = connect_db(Path(tmp) / "test.sqlite3")
        record = dict(id="q1", title="高一数学", text="根据函数图像判断最小正周期", image="q1.jpg",
                      dimensions={"subject": "数学", "grade": "高一", "knowledge_points": ["三角函数"], "difficulty": "中等"},
                      status="complete", role="question", source_image="page-01.jpg", page=1)
        index_records(db, [record])
        index_records(db, [record])
        db.commit()
        answer = search(db, "数学 高一 三角函数 不存在的关键词")
        assert len(answer["results"]) == 1, answer
        assert "不存在的关键词" in answer["results"][0]["unmatched"], answer
        natural = search(db, "给我高一数学三角函数适中难度的题")
        assert len(natural["results"]) == 1 and len(natural["results"][0]["matched"]) == 4, natural
        assert search(db, "完全不存在")["results"] == []
        assert search(db, "")["results"] == []
        assert db.execute("SELECT count(*) FROM search_index").fetchone()[0] == 1
        db.close()
    print("PASS: crop bounds; partial search; idempotent index; hard links survive move, edits and removal of export")


if __name__ == "__main__":
    check()
