import unittest
from ais_correlator import attribute_spill_to_vessels

class TestAISCorrelator(unittest.TestCase):
    
    def test_persian_gulf_attribution(self):
        """Test attribution of Persian Gulf slick to MT AL-BARAKA."""
        # Origin estimated near 25.815°N, 54.825°E
        results = attribute_spill_to_vessels(
            origin_lat=25.815,
            origin_lon=54.825,
            scenario_hint="sample_1_tanker_slick"
        )
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertEqual(top["vessel_name"], "MT AL-BARAKA")
        self.assertGreater(top["total_suspicion_score"], 80.0)
        self.assertIn("CRITICAL PRIME SUSPECT", top["attribution_tier"])
        print(f"\n[AIS ATTRIBUTION TEST] Top Suspect: {top['vessel_name']} ({top['vessel_type']})")
        print(f"[AIS ATTRIBUTION TEST] Suspicion Score: {top['total_suspicion_score']}% (Proximity: {top['score_breakdown']['proximity_score']}%, Trajectory: {top['score_breakdown']['trajectory_score']}%, Anomaly: {top['score_breakdown']['anomaly_score']}%)")
        print(f"[AIS ATTRIBUTION TEST] Red Flags: {top['anomaly_flags']}")

if __name__ == "__main__":
    unittest.main()
