# Glossary

Every abbreviation and term used in these docs and in the docstrings, with one precise line each
and a link to the page that explains it in full. Names in `code font` are columns, fields or
functions. Frame and sign rules are in [Conventions](conventions.md).

## Abbreviations

| Term | Meaning | Explained in |
|---|---|---|
| ARB | Anti-roll bar: a torsional spring that couples the left and right wheels and adds roll stiffness without adding heave stiffness. Three types are modelled: torsional, blade, beam. | [Anti-roll bar](vehicle-kinematics.md#4-anti-roll-bar) |
| FOM | Full-order model: the real kinematic solver, as opposed to the neural surrogate. | [Surrogate model](surrogate.md) |
| FVIC / SVIC | Front-view / side-view instant centre: the point about which the upright instantaneously rotates relative to the chassis, seen from the front or from the side. | [Front-view instant centre](metrics.md#front-view-instant-centre) |
| FVSA / SVSA | Front-view / side-view swing-arm length: horizontal distance from the contact patch to the FVIC / SVIC (`fvsa`, `svsa`). | [Swing-arm lengths](metrics.md#swing-arm-lengths) |
| HPO | Hyper-parameter optimisation: the Optuna search for the network architecture of the surrogate. | [Architecture search](surrogate.md#9-architecture-search-hpo-and-the-report) |
| IC | Instant centre: the instantaneous centre of rotation of the upright in a view plane, found as the intersection of the two wishbone planes cut by that plane. | [Geometry centres](metrics.md#6-geometry-centres) |
| KPI | Kingpin inclination (also steering axis inclination, SAI): lean of the steering axis seen from the front, positive when its top is inboard. | [Kingpin inclination](metrics.md#kingpin-inclination) |
| LBJ / UBJ | Lower / upper ball joint: the outer wishbone pivots on the upright (`lca_outer`, `uca_outer`). The line LBJ to UBJ is the steering (kingpin) axis. | [Hardpoints](vehicle-model.md#3-hardpoints) |
| LCA / UCA | Lower / upper control arm, i.e. the lower / upper wishbone. Each has two chassis pivots (`*_front`, `*_rear`) and one outer ball joint. | [Hardpoints](vehicle-model.md#3-hardpoints) |
| LHS | Latin hypercube sampling: stratified random sampling used to draw the design vectors of the surrogate dataset. | [Latin hypercube sampling](surrogate.md#latin-hypercube-sampling) |
| MR | Motion ratio, $-dL/dz$: shortening of the coilover per unit wheel travel. | [Motion ratio](metrics.md#motion-ratio) |
| POD | Proper orthogonal decomposition: the SVD-based compression of the output curves that the network predicts instead of the raw channels. | [Normalisation and POD](surrogate.md#6-normalisation-and-pod-compression) |
| RC | Roll centre: the point (height at the centreline) about which the body rolls relative to the ground, from the contact patch through the FVIC. | [Roll centre height](metrics.md#roll-centre-height-one-corner) |
| RCVD | *Race Car Vehicle Dynamics* (Milliken and Milliken): the textbook whose formulas for anti-features, roll stiffness and load transfer the vehicle-level metrics follow. | [Anti-features](metrics.md#7-anti-features-and-pitch-centre) |
| TLLTD | Total lateral load transfer distribution: front share of the total lateral load transfer. Higher means less front grip relative to the rear (more understeer). | [Load transfer and TLLTD](metrics.md#load-transfer-and-tlltd) |
| TPE | Tree-structured Parzen estimator: the Optuna sampler used by the architecture search. | [Architecture search](surrogate.md#9-architecture-search-hpo-and-the-report) |
| DE | Differential evolution: a population-based global optimiser, one of the drivers. | [Drivers](optimization.md#48-drivers-suspkinoptimizedrivers) |
| NSGA-II | Non-dominated sorting genetic algorithm II: the multi-objective driver used in `pareto` mode. | [Drivers](optimization.md#48-drivers-suspkinoptimizedrivers) |
| SLSQP | Sequential least-squares programming: a gradient-based local optimiser, run after DE in `de+slsqp`. | [Drivers](optimization.md#48-drivers-suspkinoptimizedrivers) |
| SAI | Steering axis inclination: another name of KPI. | [Kingpin inclination](metrics.md#kingpin-inclination) |

## Suspension and kinematics terms

| Term | Meaning | Explained in |
|---|---|---|
| Hardpoint | A named 3D point of the suspension (pivot, ball joint, rocker pickup), in metres in the ISO 8855 frame. | [Hardpoints](vehicle-model.md#3-hardpoints) |
| Corner | One wheel station (FL, FR, RL, RR) with its hardpoints. The YAML stores the left side; the right side is mirrored. | [Conventions](conventions.md#2-corners-mirroring-and-the-per-side-sign) |
| Upright | The rigid part that carries the wheel and joins the ball joints and the tie-rod outer end. | [Vehicle model](vehicle-model.md#2-theory-what-a-double-wishbone-pushrod-corner-does) |
| Wishbone | A triangular control arm with two chassis pivots and one ball joint on the upright. | [Vehicle model](vehicle-model.md#2-theory-what-a-double-wishbone-pushrod-corner-does) |
| Tie rod | The steering link between the rack end (`tie_inner`) and the upright (`tie_outer`). Its length closes the upright orientation (`psi`). | [Corner kinematics](corner-kinematics.md) |
| Rack | The steering rack; `rack` is its displacement in m (leftward positive). | [Conventions](conventions.md#51-motion-inputs) |
| Pushrod | The link from the upright or a wishbone to the rocker; carries wheel motion to the coilover. | [Vehicle model](vehicle-model.md#2-theory-what-a-double-wishbone-pushrod-corner-does) |
| Rocker (bellcrank) | The lever that pivots on the chassis and converts pushrod motion into coilover (and ARB) motion. | [Corner kinematics](corner-kinematics.md) |
| Coilover | Spring and damper unit between the rocker and the chassis; its length is `coilover_length`. | [Spring force](metrics.md#spring-force) |
| Toggle | A pushrod nearly in line with the rocker lever: zero mechanical advantage. The `toggle_margin` is the angle to that lock-up, $\pi/2$ being best. | [Corner kinematics](corner-kinematics.md) |
| `psi` | Spin of the upright about its LBJ to UBJ axis, the third unknown of the corner solve besides the two wishbone angles. | [Corner kinematics](corner-kinematics.md#1-theory) |
| Heave, roll, pitch | Body motions: vertical translation, rotation about x, rotation about y, about a configurable pivot. | [Vehicle kinematics](vehicle-kinematics.md) |
| Travel | Wheel travel along z relative to the static pose, bump positive. | [Conventions](conventions.md#51-motion-inputs) |
| Bump / droop | Wheel moving up / down relative to the body. | [Conventions](conventions.md#51-motion-inputs) |
| Contact patch | The point where the tyre meets the ground (`contact_patch`). | [Contact patch](metrics.md#32-contact-patch) |
| Sprung mass | Body mass carried by the springs; the unsprung mass moves with the wheels. | [Sprung mass](metrics.md#sprung-mass-and-sprung-cg) |
| Swing arm | The equivalent single-pivot arm of a double wishbone, from the contact patch to the IC. | [Swing-arm lengths](metrics.md#swing-arm-lengths) |
| Swing angle | Signed angle of the line from the contact patch (or wheel centre) to the IC; drives anti-features. | [Swing angles](metrics.md#swing-angles) |
| Continuation | Solving a sweep point by starting from the neighbouring converged solution. | [Corner kinematics](corner-kinematics.md) |

## Metrics terms

| Term | Meaning | Explained in |
|---|---|---|
| Camber | Wheel inclination about x relative to vertical; negative when the top leans inboard. | [Camber](metrics.md#camber) |
| Toe | Wheel yaw relative to the car axis; toe-in positive. | [Toe](metrics.md#toe) |
| Caster | Rearward lean of the steering axis seen from the side, positive when the top ball joint is behind the bottom. | [Caster](metrics.md#caster) |
| Scrub radius | Lateral distance on the ground between the contact patch and the point where the steering axis meets the ground. | [Scrub radius](metrics.md#scrub-radius) |
| Mechanical trail | Longitudinal distance on the ground from the contact patch to the steering-axis intercept, positive when the intercept is ahead of the patch. | [Mechanical trail](metrics.md#mechanical-trail) |
| Bump steer | Toe change per unit of bump, $d\,\mathrm{toe}/dz$; ideally flat. | [Bump steer](metrics.md#bump-steer) |
| Steer gain | Road-wheel angle per unit rack displacement. | [Steer gain](metrics.md#steer-gain) |
| Ackermann | Percentage of ideal Ackermann steering geometry (100 % true, 0 % parallel steer, negative anti-Ackermann). | [Ackermann](metrics.md#ackermann) |
| Anti-dive / anti-lift / anti-squat | Share of the braking or driving load transfer reacted through the suspension links instead of the springs. | [Anti-features](metrics.md#7-anti-features-and-pitch-centre) |
| Pitch centre | Intersection of the front and rear side-view lines in the xz plane. | [Pitch centre](metrics.md#pitch-centre) |
| Wheel rate | Stiffness at the wheel, $k\,\mathrm{MR}^2 + F_s\,dMR/dz$. | [Wheel rate](metrics.md#wheel-rate) |
| Ride rate | Wheel rate in series with the vertical tyre stiffness. | [Ride rate](metrics.md#ride-rate) |
| Roll stiffness | Moment per unit body roll angle resisted by an axle, N m/rad. | [Roll stiffness](metrics.md#roll-stiffness) |
| Roll gradient | Body roll per unit lateral acceleration (rad per g). | [Roll gradient](metrics.md#roll-gradient) |
| Roll axis | The line through the front and rear roll centres. | [Roll-axis height](metrics.md#roll-axis-height-at-the-cg) |
| Load transfer | Change of vertical wheel load in cornering: elastic, geometric and unsprung parts. | [Load transfer and TLLTD](metrics.md#load-transfer-and-tlltd) |
| Spindle length | Distance along the wheel axis between the steering axis and the wheel centre. | [Spindle length](metrics.md#spindle-length) |
| Kabsch | Algorithm that finds the best-fit rigid rotation between two point sets (used for body rotations). | [Joint angles](metrics.md#13-joint-angles) |

## Optimisation, surrogate and project terms

| Term | Meaning | Explained in |
|---|---|---|
| Study | The YAML that defines a design space, constraints, objectives and optimiser. | [Study YAML](../guides/study-yaml.md) |
| Constraint `g` | Normalised violation, `g <= 0` required; negative is margin. | [Optimisation](optimization.md#22-why-constraints-are-normalised) |
| Pareto mode | Multi-objective mode that returns a front of non-dominated designs instead of one weighted optimum. | [Optimisation](optimization.md#41-the-study-model-suspkinoptimizespec) |
| Manufacturing grid | The step to which hardpoint offsets are snapped (quantised) so designs are producible. | [Why quantise](optimization.md#24-why-quantise-to-a-manufacturing-grid) |
| Active learning | The surrogate loop: search on the networks, verify on the FOM, fine-tune, repeat inside a trust region. | [Active learning](surrogate.md#13-active-learning) |
| Trust region | A box around the current best design in the unit cube, whose radius grows or shrinks with agreement between network and FOM. | [Active learning](surrogate.md#13-active-learning) |
| Infill | The extra candidate designs verified by the FOM in each active-learning iteration. | [Active learning](surrogate.md#13-active-learning) |
| Oracle | A predictor that returns the exact model's curves in place of a network, used to test the surrogate path. | [Surrogate model](surrogate.md) |
| Golden snapshot | A stored result table that tests compare against, with explicit tolerances. | [Development](../guides/development.md#4-golden-tests-and-the-behaviour-lock) |
| Behaviour lock | Recorded CLI output, files and API responses that must not change unannounced. | [Development](../guides/development.md#4-golden-tests-and-the-behaviour-lock) |
| Bundle | A portable result folder of one design (tables and JSON), exchanged between runs and the web app. | [Result bundles](sweeps-and-reports.md#7-result-bundles-suspkinreportbundle) |
| ISO 8855 | The vehicle frame used everywhere: x forward, y left, z up, origin on the ground at the middle of the front axle. | [Conventions](conventions.md#1-coordinate-frame-iso-8855) |
