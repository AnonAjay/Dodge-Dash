"""Entry point:  python -m station_a --keyboard     (no camera needed)
                python -m station_a                (real camera)

TODO: parse args, pick a PoseProvider, run the loop:
      read pose -> update(state, pose, dt) -> draw(state) -> on RESULT enqueue ScoreEvent
"""
