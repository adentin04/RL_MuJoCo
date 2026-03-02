# MuJoCo API

## `mujoco.MjModel`

**For compiling the XML model.**

```python
import mujoco

model = mujoco.MjModel.from_xml_path("robot.xml")
```

The XML contains:

* Bodies
* Joints
* Geoms
* Actuators
* Sizes (nq, nv, nu, etc.)

Very commonly used fields:

```python

model.nq      # number of qpos
model.nv      # number of qvel
model.nu      # number of controls
model.nbody
model.ngeom
```


## `mujoco.MjData`

**Dynamic simulation state.**

```python
data = mujoco.MjData(model)
```

You modify/read this constantly.



### Positions & Velocities (Numpy)

```python
data.qpos     # generalized positions
data.qvel     # generalized velocities
data.qacc     # accelerations
```

### Controls

```python
data.ctrl     # control input
```

### Forces

```python
data.qfrc_applied
data.xfrc_applied
```

### Contacts

```python
data.ncon
data.contact
```

---

# Kinematics

After calling:

```python
mujoco.mj_forward(model, data)
```

You can access:

```python
data.xpos        # body world positions
data.xquat       # body orientations
data.geom_xpos
data.site_xpos
data.cvel        # body velocities
```

---


## `mujoco.mjtObj`


```python
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
```

Common values:

* `mjOBJ_BODY`
* `mjOBJ_JOINT`
* `mjOBJ_GEOM`
* `mjOBJ_SITE`
* `mjOBJ_ACTUATOR`
* `mjOBJ_SENSOR`



## `mujoco.mjtJoint`

```python
model.jnt_type[joint_id]
```

Types:

* `mjJNT_FREE`
* `mjJNT_BALL`
* `mjJNT_SLIDE`
* `mjJNT_HINGE`



## `mujoco.mjtGeom`

* `mjGEOM_BOX`
* `mjGEOM_SPHERE`
* `mjGEOM_CAPSULE`
* `mjGEOM_CYLINDER`
* `mjGEOM_MESH`
* `mjGEOM_PLANE`



# Stepping Functions

```python
mujoco.mj_step(model, data)
mujoco.mj_forward(model, data)
mujoco.mj_resetData(model, data)
```

---

# Rendering Types

If you're using the built-in viewer:

```python
import mujoco.viewer

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

Lower-level rendering types:

* `mujoco.MjvScene`
* `mujoco.MjvCamera`
* `mujoco.MjrContext`



# Model Introspection

Common arrays inside `model`:

```python
model.body_name
model.jnt_type
model.jnt_qposadr
model.jnt_dofadr
model.geom_bodyid
model.actuator_trntype
```



For RL use:

* `MjModel`
* `MjData`
* `data.qpos`
* `data.qvel`
* `data.ctrl`
* `mj_step`
* `mj_forward`
* `mj_name2id`
