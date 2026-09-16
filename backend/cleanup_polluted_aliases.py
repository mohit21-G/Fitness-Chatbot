"""
One-off maintenance: remove polluted LEARNED food aliases.

Background
----------
`save_learned_food_sync` previously stored the *raw user query* as a learned
alias with no hygiene check. Sentence fragments therefore ended up in the alias
table pointing at whatever food happened to be resolved at the time, e.g.

    "aje bhakhari"                    -> "aj amarillo yellow chile pepper"
    "did sets of reps bench press"    -> "oil, flaxseed, cold pressed"
    "clocked km trail running in minutes" -> a lemonade drink

Because fuzzy matching scores aliases too, these fragments then OUT-SCORED the
correct canonical food (e.g. "aje bhakri" scored 91 on "aje bhakhari" vs 75 on
the correct "bhakri"), producing wildly unrelated matches.

Scope / safety
--------------
* Only `alias_type == "learned"` documents are considered. Curated `primary`
  and `regional` dataset aliases are never touched — they legitimately contain
  connectives ("apple and honey sorbet", "nariyal ke saath phoolgobhi").
* Only aliases containing a *stop token* (pronoun / eating verb / time-meal
  context) are removed — i.e. provable sentence fragments. Long-but-legitimate
  names (e.g. "turkey, ground, 85% lean, 15% fat, raw") are kept.

Run:  python cleanup_polluted_aliases.py           (dry run, shows what it would do)
      python cleanup_polluted_aliases.py --apply   (performs the deletion)
"""
import re
import sys
import io

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from database import sync_db as db
from food_repository import _ALIAS_STOP_TOKENS


def is_sentence_fragment(alias: str) -> bool:
    """True when the alias provably contains a non-food stop token."""
    tokens = re.findall(r"[^\W\d_]+", (alias or "").lower())
    return any(t in _ALIAS_STOP_TOKENS for t in tokens)


def main() -> None:
    apply_changes = "--apply" in sys.argv

    learned = list(db.food_aliases.find(
        {"alias_type": "learned"}, {"alias": 1, "food_id": 1, "_id": 0}
    ))
    targets = [d for d in learned if is_sentence_fragment(d.get("alias", ""))]

    print("=" * 70)
    print("  Polluted learned-alias cleanup")
    print("=" * 70)
    print(f"total aliases        : {db.food_aliases.count_documents({})}")
    print(f"learned aliases      : {len(learned)}")
    print(f"sentence fragments   : {len(targets)}")
    print()

    for d in targets:
        food = db.foods.find_one({"food_id": d["food_id"]}, {"food_name": 1, "_id": 0}) or {}
        print(f"  REMOVE alias={d['alias']!r:44} -> {food.get('food_name')!r}")

    if not targets:
        print("Nothing to clean.")
        return

    if not apply_changes:
        print("\n[DRY RUN] Re-run with --apply to delete the aliases listed above.")
        return

    removed = 0
    for d in targets:
        res = db.food_aliases.delete_many({"alias": d["alias"], "alias_type": "learned"})
        removed += res.deleted_count
    print(f"\n[DONE] Removed {removed} polluted learned aliases.")


if __name__ == "__main__":
    main()
