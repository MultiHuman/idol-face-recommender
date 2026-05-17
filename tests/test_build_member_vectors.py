from __future__ import annotations

import unittest

from src.build_member_vectors import apply_gender_overrides, resolve_group_gender_overrides


class BuildMemberVectorsTest(unittest.TestCase):
    def test_coed_group_override_preserves_member_level_gender(self) -> None:
        rows = [
            {"member_id": "coed__female", "group_name": "COED GROUP", "gender": "F"},
            {"member_id": "coed__male", "group_name": "COED GROUP", "gender": "M"},
        ]

        overrides = resolve_group_gender_overrides(
            rows_to_write=rows,
            external_gender={"COED GROUP": "COED"},
        )

        self.assertNotIn("COED GROUP", overrides)

    def test_binary_group_override_still_applies(self) -> None:
        rows = [
            {"member_id": "group__one", "group_name": "GIRL GROUP", "gender": "M"},
            {"member_id": "group__two", "group_name": "GIRL GROUP", "gender": "M"},
        ]

        overrides = resolve_group_gender_overrides(
            rows_to_write=rows,
            external_gender={"GIRL GROUP": "F"},
        )

        self.assertEqual(overrides["GIRL GROUP"], "F")

    def test_member_gender_override_wins_over_group_override(self) -> None:
        rows = [{"member_id": "coed__male", "group_name": "COED GROUP", "gender": "F"}]

        apply_gender_overrides(
            rows_to_write=rows,
            group_gender={"COED GROUP": "F"},
            member_gender={"coed__male": "M"},
        )

        self.assertEqual(rows[0]["gender"], "M")


if __name__ == "__main__":
    unittest.main()
