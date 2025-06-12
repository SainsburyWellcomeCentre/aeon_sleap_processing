# Fight Detection Logic

## Step 1: Identifying Candidate Fight Frames
Frames are initially selected based on three conditions:
1. **Proximity:**
    - The centroids of the two mice must be within **20 px** of each other.
2. **Skeleton Plausibility:**
    - When mice overlap or are in chaotic poses (as during a fight), SLEAP’s keypoint tracking can produce implausible skeletons.
    - A frame is flagged if either:
        - A mouse’s own head-to-nose distance exceeds **7 px**, **or**
        - The mean distance between a mouse’s spine points exceeds **10 px**.
3. **Blob Movement:**
    - The Bonsai-tracked blob (which tracks the combined movement of both mice) must have a speed greater than **3 cm/s**.
    - This low threshold is used because the blob can encompass both mice and is less reliable at distinguishing individual movement.

Frames meeting **all three conditions** are marked as potential fight frames. These frames are then grouped into fight bouts:
- **Grouping Rule:**
    - Frames are considered part of the same fight if they occur within **200 frames** of one another.
    - Only groups (subarrays) with more than **5 frames** are retained for further analysis.

## Step 2: considering empty frames
During intense fights, SLEAP’s keypoint tracking can fail entirely, leading to **empty frames** rather than implausible skeletons. Because these empty frames can still represent fighting behavior, we handle them as follows:
1. **Selecting Empty Frames**
    - We first identify empty frames in which the mice were known to be **close to each other** in the previously detected frame (fulfilling the proximity requirement from Step 1).
2. **Regrouping with a Stricter Threshold**
    - We **merge** these selected empty frames with the fight frames identified in Step 1 and regroup them into bouts if they occur within **100 frames** of one another.
    - This **100-frame** gap is stricter than the **200-frame** rule from Step 1, acknowledging that empty frames are more permissive than the checks from Step 1
3. **Discarding Isolated Empty Frames**
    - Any subarray composed **entirely** of empty frames (i.e., without any frames from Step 1) is **discarded**, as empty frames on their own are not enough evidence that a fight is occurring

By merging these empty frames with previously flagged fight frames, we **extend** or **connect** potential fight bouts that might otherwise be split by tracking failures. This step increases the likelihood of capturing genuine continuous fights even when SLEAP loses track of the mice.

## Step 3: checking individual mouse speeds
In Step 1, the **low Bonsai blob speed threshold** effectively removed frames where the mice were **completely stationary** (e.g., sleeping), but it does not capture the **high-speed movements** typical of fighting. Therefore, we apply a more detailed **speed check** here, focusing on individual mouse speeds. Because this step requires **identity cleanup** to avoid mouse swaps, it is **computationally expensive**, so it is only performed on **pre-filtered fight bouts** from Steps 1 and 2.
1. **Identity Cleanup**
    - We compare each mouse’s position from frame to frame to detect and correct any **identity swaps**, ensuring that each mouse’s speeds are computed accurately.
2. **Speed Threshold**
    - After resolving identities, we compute each mouse’s **mean speed** over the bout.
    - A fight is **confirmed** if either:
        - **One mouse** exceeds **20 cm/s**, **or**
        - **Both mice** exceed **15 cm/s** on average

## Step 4: Enforcing Minimum Fight Duration
To remove brief, likely spurious detections:
1. **Duration Check:**
    - Only fights lasting at least **1 second** are retained.