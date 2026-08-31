"""
Regression tests for Issue #207: Video generation FPS handling

These tests verify that requested FPS values are properly passed through
the video generation pipeline and affect the final output.
"""

import os
import tempfile

import pytest
from PIL import Image


# Test the _save_video function directly
def test_save_video_honours_fps():
    """Test that _save_video uses the provided FPS to set GIF frame duration."""
    from model.serve import _save_video

    # Create test frames with different colors to avoid optimization
    frames = []
    for i in range(3):
        # Create a simple test image with different colors
        color = (i * 80, 0, 0)
        frames.append(Image.new('RGB', (64, 64), color=color))

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test that different FPS values produce different durations
        # Focus on values where PIL behavior is predictable
        fps_values = [10, 20, 25]
        durations = []

        for fps in fps_values:
            filename = _save_video(frames, temp_dir, fps=fps)
            filepath = os.path.join(temp_dir, filename)

            # Verify the file was created
            assert os.path.exists(filepath), f"Video file not created for FPS {fps}"

            # Load the GIF and check its frame duration
            with Image.open(filepath) as gif:
                actual_duration = gif.info.get('duration', None)
                assert actual_duration is not None, f"No duration info for FPS {fps}"
                durations.append((fps, actual_duration))

        # Verify that higher FPS produces shorter duration (inverse relationship)
        for i in range(len(durations) - 1):
            fps1, duration1 = durations[i]
            fps2, duration2 = durations[i + 1]

            # Higher FPS should have shorter duration
            if fps2 > fps1:
                assert duration2 <= duration1, f"Higher FPS {fps2} should have shorter duration than {fps1}, got {duration2}ms vs {duration1}ms"

        # Test specific known good case
        filename_10fps = _save_video(frames, temp_dir, fps=10)
        filepath_10fps = os.path.join(temp_dir, filename_10fps)
        with Image.open(filepath_10fps) as gif:
            duration_10fps = gif.info.get('duration')
            # GIF stores duration in hundredths of a second (10ms units)
            # 1000/10 = 100ms = 10 hundredths exactly
            expected_duration_10fps = round(1000 / 10 / 10) * 10  # 100ms for 10 FPS
            assert duration_10fps == expected_duration_10fps, f"10 FPS should produce {expected_duration_10fps}ms duration, got {duration_10fps}ms"


def test_save_video_default_fps():
    """Test that _save_video uses default 24 FPS when fps=None."""
    from model.serve import DEFAULT_VIDEO_FPS, _save_video

    # Create frames with different colors
    frames = [Image.new('RGB', (32, 32), color=(255, 0, 0)), Image.new('RGB', (32, 32), color=(0, 255, 0))]

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = _save_video(frames, temp_dir)  # No fps parameter
        filepath = os.path.join(temp_dir, filename)

        with Image.open(filepath) as gif:
            actual_duration = gif.info.get('duration')
            # For 24 FPS, duration should be approximately 1000/24 ≈ 41.67ms
            # GIF stores duration in hundredths of a second (10ms units)
            # So 41.67ms rounds to 4 hundredths = 40ms
            expected_duration = round(1000 / DEFAULT_VIDEO_FPS / 10) * 10
            assert actual_duration == expected_duration, f"Expected {expected_duration}ms default duration (for {DEFAULT_VIDEO_FPS} FPS, GIF quantized to 10ms), got {actual_duration}ms"


def test_save_video_zero_fps_raises():
    """Test that _save_video raises ValueError when fps <= 0."""
    from model.serve import _save_video

    # Create frames with different colors
    frames = [Image.new('RGB', (32, 32), color=(0, 255, 0)), Image.new('RGB', (32, 32), color=(0, 0, 255))]

    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError, match="fps must be positive"):
            _save_video(frames, temp_dir, fps=0)


def test_video_generation_end_to_end_fps():
    """Test that FPS parameter flows through the complete video generation pipeline."""
    # This test requires working model dependencies, but we can test the serve layer directly
    try:
        import os
        import tempfile

        from model.serve import handle

        # Create a mock generator that returns predictable results with different frames
        class MockGenerator:
            def generate_video(self, prompt, **kwargs):
                # Simulate returning frames with different colors so they don't get optimized away
                num_frames = kwargs.get('num_frames', 4)
                frames = []
                for i in range(num_frames):
                    color = (i * 50, 0, 0)  # Different red shades
                    frames.append(Image.new('RGB', (64, 64), color=color))

                # Create a mock request object that has the fps attribute
                class MockRequest:
                    def __init__(self, fps):
                        self.fps = fps
                    def to_dict(self):
                        return {"width": 64, "height": 64}

                # Make sure to pass through the fps from kwargs
                fps = kwargs.get('fps', 24)
                return frames, MockRequest(fps)

        generator = MockGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test that different FPS values are handled correctly
            test_cases = [12, 24, 30]
            for fps in test_cases:
                result = handle(
                    generator,
                    "video",
                    {"prompt": "test video", "fps": fps, "num_frames": 4, "out_dir": temp_dir}
                )

                assert "fps" in result, f"FPS not returned in result for fps={fps}"
                assert result["fps"] == fps, f"Expected fps={fps}, got {result['fps']}"
                assert "file" in result, f"No file returned for fps={fps}"

                # Verify the actual file has correct frame duration
                # GIF stores duration in hundredths of a second (10ms units)
                filepath = os.path.join(temp_dir, result["file"])
                with Image.open(filepath) as gif:
                    expected_duration = round(1000 / fps / 10) * 10
                    actual_duration = gif.info.get('duration')
                    assert actual_duration == expected_duration, \
                        f"FPS {fps}: expected {expected_duration}ms (GIF quantized to 10ms), got {actual_duration}ms"

    except ImportError:
        pytest.skip("Model dependencies not available for end-to-end test")


if __name__ == "__main__":
    # Run tests directly
    test_save_video_honours_fps()
    test_save_video_default_fps()
    test_save_video_zero_fps_raises()
    print("All FPS regression tests passed!")
