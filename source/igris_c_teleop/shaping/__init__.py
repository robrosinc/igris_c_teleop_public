"""Reference shaping between retargeting and every output: the convergence ramp from the
robot's measured pose (``converge``), the per-arm hold on hand-tracking loss (``arm_hold``)
and the ramp primitives they share (``ramp``). Safety-relevant: these are the only rate
limits between a Start/Resume and the robot's joint targets.
"""
