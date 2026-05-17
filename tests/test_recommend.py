from __future__ import annotations

import unittest

import numpy as np

from src.recommend import MemberVector, recommend_from_members


def _member(
    member_id: str,
    group_name: str,
    vector: list[float],
    *,
    image_count: int = 5,
    confidence: float = 0.8,
    gender: str = "F",
) -> MemberVector:
    return MemberVector(
        member_id=member_id,
        group_name=group_name,
        member_name=member_id,
        image_count=image_count,
        confidence=confidence,
        gender=gender,
        vector=np.asarray(vector, dtype=np.float32),
    )


class RecommendFromMembersTest(unittest.TestCase):
    def test_quality_filters_apply_to_candidates_not_liked_members(self) -> None:
        members = [
            _member("liked", "A", [1.0, 0.0, 0.0], image_count=1, confidence=0.1),
            _member("stable", "B", [0.95, 0.05, 0.0], image_count=5, confidence=0.8),
            _member("few_images", "C", [0.9, 0.1, 0.0], image_count=4, confidence=0.8),
            _member("low_confidence", "D", [0.85, 0.15, 0.0], image_count=5, confidence=0.2),
        ]

        rows = recommend_from_members(
            members,
            liked_member_ids=["liked"],
            top_k=10,
            gender_filter="off",
            min_image_count=5,
            min_confidence=0.35,
        )

        self.assertEqual([row["member_id"] for row in rows], ["stable"])

    def test_auto_gender_filter_uses_selected_member_gender(self) -> None:
        members = [
            _member("liked", "A", [1.0, 0.0, 0.0], gender="F"),
            _member("female_candidate", "B", [0.9, 0.1, 0.0], gender="F"),
            _member("male_candidate", "C", [0.95, 0.05, 0.0], gender="M"),
        ]

        rows = recommend_from_members(
            members,
            liked_member_ids=["liked"],
            top_k=10,
            gender_filter="auto",
        )

        self.assertEqual([row["member_id"] for row in rows], ["female_candidate"])

    def test_group_cap_allows_next_group_candidate(self) -> None:
        members = [
            _member("liked", "A", [1.0, 0.0, 0.0]),
            _member("same_group_best", "B", [0.99, 0.01, 0.0]),
            _member("same_group_second", "B", [0.98, 0.02, 0.0]),
            _member("other_group", "C", [0.7, 0.3, 0.0]),
        ]

        rows = recommend_from_members(
            members,
            liked_member_ids=["liked"],
            top_k=2,
            gender_filter="off",
            max_per_group=1,
        )

        self.assertEqual([row["member_id"] for row in rows], ["same_group_best", "other_group"])

    def test_member_aliases_dedupe_cross_group_candidates(self) -> None:
        members = [
            _member("liked", "A", [1.0, 0.0, 0.0]),
            _member("alias_best", "B", [0.99, 0.01, 0.0]),
            _member("alias_second", "C", [0.98, 0.02, 0.0]),
            _member("other", "D", [0.7, 0.3, 0.0]),
        ]

        rows = recommend_from_members(
            members,
            liked_member_ids=["liked"],
            top_k=3,
            gender_filter="off",
            member_aliases={
                "alias_best": "same_person",
                "alias_second": "same_person",
            },
        )

        self.assertEqual([row["member_id"] for row in rows], ["alias_best", "other"])

    def test_member_alias_matching_liked_member_is_excluded(self) -> None:
        members = [
            _member("gfriend__eunha", "GFRIEND", [1.0, 0.0, 0.0]),
            _member("viviz__eunha", "VIVIZ", [0.99, 0.01, 0.0]),
            _member("other", "D", [0.8, 0.2, 0.0]),
        ]

        rows = recommend_from_members(
            members,
            liked_member_ids=["gfriend__eunha"],
            top_k=3,
            gender_filter="off",
            member_aliases={
                "gfriend__eunha": "jung_eunbi",
                "viviz__eunha": "jung_eunbi",
            },
        )

        self.assertEqual([row["member_id"] for row in rows], ["other"])


if __name__ == "__main__":
    unittest.main()
