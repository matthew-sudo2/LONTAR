import unittest

from src.ingestion import build_query
from src.phase2 import filter_records
from src.relevance import evaluate_record_relevance, triage_records


class RelevanceGuardrailTests(unittest.TestCase):
    def test_health_query_expands_aliases_without_bare_treatment(self):
        query = build_query(["macadamia"], focus="Health & Clinical Therapy")

        self.assertIn("macadamia oil", query)
        self.assertIn("cholesterol", query)
        self.assertNotIn(" OR treatment", query)

    def test_rejects_macadamia_shell_materials_for_health_focus(self):
        record = {
            "title": "Removal of tetracycline by NaOH-activated carbon produced from macadamia nut shells",
            "year": 2014,
            "citation_count": 778,
            "abstract": (
                "Activated carbon prepared from macadamia nut shells was used for adsorption "
                "of tetracycline from wastewater and aqueous solution."
            ),
        }

        relevance = evaluate_record_relevance(
            record,
            ingredients=["macadamia"],
            focus="Health & Clinical Therapy",
        )

        self.assertEqual(relevance["relevance_status"], "rejected")
        self.assertIn(
            "environmental/materials context conflicts with health focus",
            relevance["relevance_reject_reasons"],
        )

    def test_accepts_macadamia_nut_lipid_health_record(self):
        record = {
            "title": "Macadamia nut diet improves cholesterol and serum lipid profiles",
            "year": 2020,
            "citation_count": 12,
            "abstract": (
                "A human dietary intervention studied macadamia nuts, cholesterol, "
                "serum lipids, and cardiovascular health outcomes."
            ),
        }

        relevance = evaluate_record_relevance(
            record,
            ingredients=["macadamia"],
            focus="Health & Clinical Therapy",
        )

        self.assertEqual(relevance["relevance_status"], "accepted")
        self.assertGreaterEqual(relevance["relevance_score"], 45)

    def test_phase2_filters_and_sorts_by_relevance(self):
        bad_record = {
            "title": "Nanomaterials from macadamia nutshells for wastewater treatment",
            "year": 2026,
            "citation_count": None,
            "abstract": "Macadamia nutshell carbon dots adsorb pollutants in wastewater treatment.",
        }
        good_record = {
            "title": "Macadamia oil and cardiovascular nutrition",
            "year": 2021,
            "citation_count": 5,
            "abstract": "Macadamia oil contains dietary lipids studied for cardiovascular health.",
        }

        filtered = filter_records(
            [bad_record, good_record],
            min_citations=0,
            recent_year=2018,
            ingredients=["macadamia"],
            focus="Health & Clinical Therapy",
        )

        self.assertEqual([record["title"] for record in filtered], [good_record["title"]])
        self.assertIn("relevance_score", filtered[0])

    def test_triage_keeps_rejection_reasons_for_audit(self):
        accepted, rejected = triage_records(
            [
                {
                    "title": "Macadamia oil nutrition",
                    "abstract": "Dietary macadamia oil and lipid health effects.",
                },
                {
                    "title": "Macadamia shell activated carbon",
                    "abstract": "Activated carbon adsorption for wastewater pollutants.",
                },
            ],
            ingredients=["macadamia"],
            focus="Health & Clinical Therapy",
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 1)
        self.assertTrue(rejected[0]["relevance_reject_reasons"])


if __name__ == "__main__":
    unittest.main()
