# 🎸 ITC6109 – Machine Vision Final Project

This project performs guitar body and head shape matching using classical computer vision techniques on curated guitar image data.

---

## 📁 Project Structure

```plaintext
.
├── images/                             # Full guitar images organized by guitar type
│   ├── img-double_cut_guitars/
│   │   ├── details/                    # Subcomponents and scripts for double cuts
│   │   │   ├── double_cut_temp.jpg     # Reference template for double cuts
│   │   │   ├── img_sample/             # Raw examples for visualization
│   │   │   ├── mapping/                # Mapping helper resources
│   │   │   └── double_cuts.py          # Scraper of the double-cut guitar product category listings on thomann.de
│   ├── img-jazz_guitars/
│   ├── img-single_cut_guitars/
│   ├── img-st_guitars/
│   └── img-tele_guitars/

├── report/
│   ├── ITC6109_Final_Report_...pdf     # Final project report
│   └── ITC6109_PP_...pptx              # Presentation slides

├── template_matching_code/            # All matching approaches/scripts
│   ├── naive_approach_matching_examples/
│   │   ├── matched_*.png               # Example visual results
│   │   └── matched_result.jpg
│   ├── template_matching_code/
│   │   ├── 1_exploratory.py            # Early EDA and testing
│   │   ├── 2_template_creation_new.py  # New template creation logic
│   │   ├── 3_template_matching.py      # Basic matching logic
│   │   ├── 3.2_template_matching_...   # Naive matching w/ 2D rot/scaling
│   │   ├── 3.5_template_matching_...   # Naive matching w/ 3D rot/scaling
│   │   ├── 4_guitar_part_position_...  # Identifies head/body positions
│   │   ├── 5_new_template_idea_...     # New orientation-aware templates
│   │   ├── 6_new_template_creation.py  # Final version of template creation
│   │   ├── 7_new_template_matching.py  # Final version of matching
│   │   └── 8_Final_Code.py             # Main script for reproducing final results
│   ├── TRIAL_key_point_matching_...    # Abandoned keypoint-based trials
│   └── done.png                        # Flag/output indicator

├── templates/                         # Saved template .npy arrays
│   ├── *_head_template.png.npy
│   ├── *_upper_body_template.png.npy
│   └── full_example_on_tele.py         # One-shot template match on Telecaster

├── test_images/                       # Evaluation/test set

├── vertical_lines_tool.py             # (Unused) vertical line checker/debugger
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
