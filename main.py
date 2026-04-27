import sys

import cv2

from traffic_system.gui import launch_app
from traffic_system.pipeline import EngineConfig, TrafficViolationEngine
from traffic_system.settings import DEFAULT_SAMPLE_IMAGE


def main() -> None:
    if "--check" in sys.argv:
        engine = TrafficViolationEngine()
        print("\n".join(engine.capability_messages()))
        return

    if "--smoke-test" in sys.argv:
        engine = TrafficViolationEngine()
        frame = cv2.imread(str(DEFAULT_SAMPLE_IMAGE))
        if frame is None:
            raise FileNotFoundError(f"Sample image not found: {DEFAULT_SAMPLE_IMAGE}")
        result = engine.analyze_frame(frame, 0, EngineConfig())
        print(f"violations={len(result.violations)}")
        print(f"traffic_light_state={result.traffic_light_state}")
        return

    launch_app()


if __name__ == "__main__":
    main()
