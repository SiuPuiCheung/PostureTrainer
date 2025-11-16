"""Sample unit tests for pose analysis functions."""

import pytest
import numpy as np
from src.utils.geometry import calculate_angle, get_point
from src.core.pose_analysis import front_angle_analysis


class MockLandmark:
    """Mock landmark for testing."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


class TestGeometry:
    """Test geometry utility functions."""
    
    def test_calculate_angle_right_angle(self):
        """Test calculation of a 90-degree angle."""
        a = (0, 0)
        b = (0, 1)
        c = (1, 1)
        angle = calculate_angle(a, b, c)
        assert abs(angle - 90.0) < 0.1
    
    def test_calculate_angle_straight(self):
        """Test calculation of a 180-degree angle."""
        a = (0, 0)
        b = (1, 0)
        c = (2, 0)
        angle = calculate_angle(a, b, c)
        assert abs(angle - 180.0) < 0.1
    
    def test_get_point(self):
        """Test landmark point extraction."""
        class MockLandmarkEnum:
            value = 0
        
        landmarks = [MockLandmark(0.5, 0.7)]
        point = get_point(landmarks, MockLandmarkEnum())
        assert point == (0.5, 0.7)


class TestConfig:
    """Test configuration loading."""
    
    def test_config_loads(self):
        """Test that config loads without errors."""
        from src.utils.config_loader import Config
        config = Config()
        assert config is not None
        assert 'output_dir' in config.paths
        assert len(config.analysis_types) == 6
    
    def test_body_labels_by_index(self):
        """Test retrieving body labels by index."""
        from src.utils.config_loader import Config
        config = Config()
        labels = config.get_body_labels_by_index(0)
        assert len(labels) > 0
        assert isinstance(labels, list)


class TestPoseAnalysis:
    """Test pose analysis functions."""
    
    def test_front_angle_analysis_returns_tuple(self):
        """Test that front_angle_analysis returns correct tuple length."""
        # Create mock landmarks (33 landmarks as per MediaPipe)
        landmarks = [MockLandmark(i * 0.01, i * 0.02) for i in range(33)]
        
        # Mock mp_pose (not used in calculation, just passed through)
        class MockMpPose:
            pass
        
        result = front_angle_analysis(landmarks, MockMpPose())
        assert isinstance(result, tuple)
        assert len(result) == 15  # Expected number of outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
