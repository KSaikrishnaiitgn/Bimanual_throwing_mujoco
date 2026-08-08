# VelGrasp

This repository contains the framework, mathematics, and codebase for performing **dynamic tasks using two heterogeneous robots** under **velocity control**.

A **second-order Dynamical System (DS)** is used to generate the desired object acceleration required for throwing the object using two coordinated robots.  
The computed object acceleration acts as a **feedforward term** inside the **object-level impedance control equation** (refer to Eq. 13 and Eq. 14 in the ICRA paper).

---

## 🔧 Loading the Model

Use the **`test.py`** file inside the **`scripts/`** folder to load the Mujoco scene consisting of:

- Two Heal robots  
- A box between them  
- Predefined grasping sites for dual-arm manipulation  

The main Mujoco XML file is **`scene.xml`**, which defines both robots, the object, and the grasp sites.

You can also find the XML description of a **single Heal robot** inside:


---

## 🤖 Inverse Kinematics (IK)

Inside **`scripts/mink IK/`**, you will find:

- Dual-arm IK scripts  
- Single-arm IK scripts  

These allow the robots to reach grasp sites on the box, starting from pre-grasp configurations.

---

## 🚀 Dynamic Task Implementation

Users can write their **DS-based dynamic task** implementations inside the:



folder.

---

# ⚠️ IMPORTANT  
# 🚀 How to Push Your Work to the `dev` Branch (NOT main)

Please **do NOT push to `main`**.  
All user contributions must go to **`dev`**.

Use the following commands EXACTLY:

```bash
# 1. Clone the repository
git clone git@github.com:Samriddhi-dubey06/samriddhi_mujoco_bimanual_ds_dynamic_task.git

# 2. Enter the project directory
cd samriddhi_mujoco_bimanual_ds_dynamic_task

# 3. Fetch all remote branches
git fetch origin

# 4. Switch to the dev branch (IMPORTANT)
git checkout dev

# 5. Make your changes
# ---- edit files normally ----

# 6. Stage your changes
git add .

# 7. Commit your changes
git commit -m "Describe what you changed"

# 8. Push to the dev branch
git push origin dev

# (If this is your first time pushing to dev)
git push -u origin dev
