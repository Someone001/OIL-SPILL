import unittest
import datetime
from drift_tracer import trace_backward_drift, compute_drift_vector

class TestDriftTracer(unittest.TestCase):
    
    def test_drift_vector_physics(self):
        """Test Coriolis deflection and wind leeway factor."""
        # Northern hemisphere: lat = 25.0
        # Current = 0.3 m/s East (u=0.3, v=0)
        # Wind = 10 m/s North (u=0, v=10)
        u_d, v_d, spd_kts, hdg = compute_drift_vector(
            u_curr=0.3, v_curr=0.0,
            u_wind=0.0, v_wind=10.0,
            lat=25.0,
            wind_factor=0.03,
            coriolis_angle_deg=20.0
        )
        # Leeway should add 0.3 m/s rotated +20° to the right (East)
        self.assertGreater(u_d, 0.3)
        self.assertGreater(spd_kts, 0.5)
        print(f"\n[DRIFT PHYSICS TEST] Total drift speed: {spd_kts} knots, Heading: {hdg}°")
        
    def test_24hr_backward_trace(self):
        """Test full 24-hour backward trajectory integration."""
        result = trace_backward_drift(
            detection_lat=25.82,
            detection_lon=55.12,
            hours_back=24
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["hourly_trajectory"]), 25) # T=0 to T=24
        self.assertGreater(result["estimated_origin"]["total_drift_distance_km"], 5.0)
        self.assertGreater(result["estimated_origin"]["uncertainty_radius_km"], 1.0)
        print(f"[24HR TRACE TEST] Origin Lat/Lon: {result['estimated_origin']['lat']}, {result['estimated_origin']['lon']}")
        print(f"[24HR TRACE TEST] Total Distance: {result['estimated_origin']['total_drift_distance_km']} km, Uncertainty: ±{result['estimated_origin']['uncertainty_radius_km']} km")

if __name__ == "__main__":
    unittest.main()
