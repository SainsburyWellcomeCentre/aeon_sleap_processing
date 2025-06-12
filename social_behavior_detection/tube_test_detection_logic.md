# Tube Test Detection Logic

## Step 1: Identifying Candidate Tube Test Start Frames
Frames are initially selected based on four conditions:
1. **Opposite Orientations:**
    - The mice must have opposite orientations (within a tolerance of **45°**).
2. **Proximity:**
    - The distance between the mice's centroids must be less than **50 px**, ensuring they are close.
3. **Asymmetrical Distance Check:**
    - For Mouse A, the distance from its spine to its own head should be smaller than the distance from its spine to Mouse B’s head. This condition helps remove cases where the mice are side-by-side.
    - _See illustration:_ ![Image 1](tube_test_detection_logic_images/Pasted%20image%2020250320170416.png)![Image 2](tube_test_detection_logic_images/Pasted%20image%2020250320170429.png)
4. **Tail-to-Tail vs. Nose-to-Nose:**
    - The tail-to-tail distance should be greater than the nose-to-nose distance to eliminate cases where the mice are back-to-back.

Frames meeting **all four conditions** are marked as candidate tube test start frames. These frames are further filtered to retain only those where both mice are within the corridor ROI and not adjacent to any openings (to the main arena or nest).
Next, they are grouped into separate candidate tube test start events
- **Grouping Rule:**
    - Frames are considered part of the same fight if they occur within **20 frames** of one another.
    - Only groups (subarrays) with more than **15 frames** are retained for further analysis.

## Step 2: Checking for Tracking Errors
When the mice are close together, SLEAP key point tracking becomes more error prone. Each candidate tube test start event is therefore checked for tracking errors, particularly **skeleton flipping** .e., tail end being mistaken for head, which leads to false candidate tube test start detections. 
- If at any time point within a candidate tube test start event from Step 1 both mice exhibit the same orientation, the event is discarded

## Step 3: Identifying Tube Test End Frames
Once the tube test start conditions are broken, it indicates that one of the mice may have turned around—marking a true tube test event—or that another behavior (e.g., climbing, fighting) has occurred, in which case the candidate event should be discarded. For each candidate event, a **1-second window** is established from the last tube test start frame to locate potential end frames. Within this window, frames must satisfy three conditions:
1. **Same Orientation:**  
    - The mice exhibit the same orientation (within a tolerance of **45°**).
2. **Sufficient Separation:**  
    - The distance between the mice's centroids must be larger than **30px**, thereby removing frames where the mice are fighting or remain side-by-side.
3. **Consistency Check:**  
    - The distance between the mice's centroids must be smaller than 60px, which helps eliminate frames with “teleportation” artifacts from tracking errors.

## Step 4: Confirming Tube Test End Frames
If any tube test end frames are detected within a candidate event that meet the conditions above, additional verifications are conducted. Because this step requires **identity cleanup** to avoid mouse swaps, it is **computationally expensive**, so it is only performed on **pre-filtered tube test events** from steps 1, 2 and 3. 
1. **Identity Cleanup**
    - We compare each mouse’s position from frame to frame to detect and correct any **identity swaps**
2. **Determining the Loser:**
	- The mouse that turns around (the “loser”) is identified. To avoid false positives (e.g., when Mouse A squeezes past Mouse B, causing Mouse B to turn around), it is verified that the loser is positioned in front of the winner. This is determined by comparing distances:
	    - The loser is considered “in front” if the distance from the loser’s head to the winner’s tail is greater than the distance from the loser’s tail to the winner’s head.
	- _See illustration:_ ![Image 3](tube_test_detection_logic_images/Pasted%20image%2020250320172624.png)
3. **Movement Threshold:**
	- The loser’s average movement during the 1-second window must exceed **2 px/s**. This criterion helps to exclude cases where a mouse appears to turn around but remains stationary (e.g., during grooming).