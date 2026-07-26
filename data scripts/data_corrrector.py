import os
import cv2
import glob

def rapid_manual_unification(input_dir, output_dir):
    """
    A high-speed manual triage tool with a fixed UI scale and Undo feature
    to guarantee 100% laterality unification.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Grab all images and sort them so they appear in a predictable order
    image_paths = sorted(glob.glob(os.path.join(input_dir, '*.[jp][pn]*[g]')))
    total_images = len(image_paths)
    
    print(f"[*] Starting Rapid Triage for {total_images} images.")
    print("\n=== CONTROLS ===")
    print("[ A ] - LEFT EYE (Script will Flip it)")
    print("[ D ] - RIGHT EYE (Script will Keep it)")
    print("[ Z ] - UNDO MISTAKE (Goes back one image)")
    print("[ Q ] - SAVE & QUIT")
    print("================\n")

    session_history = []

    cv2.namedWindow("Rapid Sorter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Rapid Sorter", 800, 800)

    i = 0
    while i < total_images:
        path = image_paths[i]
        filename = os.path.basename(path)
        save_path = os.path.join(output_dir, filename)
        
        # Skip if already processed in a previous session
        if os.path.exists(save_path) and path not in session_history:
            i += 1
            continue

        img = cv2.imread(path)
        if img is None:
            i += 1
            continue

        # --- Fixed UI Display Scaling ---
        # Resize image to a standard 800x800 UI canvas first so text proportions stay consistent
        display_img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA)
        
        # Draw a clean, well-fitted background box for text readability
        cv2.rectangle(display_img, (15, 15), (785, 115), (0, 0, 0), -1)
        
        # Draw Progress Text
        progress_text = f"Image: {i + 1} / {total_images}"
        cv2.putText(display_img, progress_text, (30, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Draw Controls Text (fully visible now)
        controls_text = "[A] Flip | [D] Keep | [Z] Undo | [Q] Quit"
        cv2.putText(display_img, controls_text, (30, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # ------------------------------------

        cv2.imshow("Rapid Sorter", display_img)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('a'):  # Left Eye -> Flip
            final_img = cv2.flip(img, 1)
            cv2.imwrite(save_path, final_img)
            session_history.append(path)
            i += 1
            
        elif key == ord('d'):  # Right Eye -> Keep
            cv2.imwrite(save_path, img)
            session_history.append(path)
            i += 1
            
        elif key == ord('z'):  # UNDO MISTAKE
            if len(session_history) > 0:
                last_path = session_history.pop()
                last_save_path = os.path.join(output_dir, os.path.basename(last_path))
                
                if os.path.exists(last_save_path):
                    os.remove(last_save_path)
                    
                i -= 1 
                print(f"[*] Undo triggered. Going back to {os.path.basename(last_path)}")
            else:
                print("[!] Cannot undo. You are at the beginning of this session.")
                
        elif key == ord('q'):  # Quit
            print(f"\n[*] Pausing triage. You can safely restart the script later.")
            break

    cv2.destroyAllWindows()
    print("\n[*] Sorting session ended.")

if __name__ == "__main__":
    INPUT_DATASET = "../Raw_datasets/Combined_DR_Dataset/normalized_images"
    OUTPUT_DATASET = "../Raw_datasets/Combined_DR_Dataset/normalized_images_unified"
    
    rapid_manual_unification(INPUT_DATASET, OUTPUT_DATASET)