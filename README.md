# 🎸 ITC6109 – Machine Vision Final Project
**Course:** ITC6109 – Machine Vision

**Institution:** MSc in Data Science, The American College of Greece

**Term:** Fall 2024

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
│   │   └── TRIAL_key_point_matching_with_2D_rot.py  # Keypoint-based (failed) approach 
│   │   └── TRIAL_key_point_matching_with_3D_rot.py  # Another failed attempt using features
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
│   ├── *_head_template.png.npy          # Head cutouts per guitar type
│   ├── *_upper_body_template.png.npy    # Body cutouts per guitar type
│   └── full_example_on_tele.py         # One-shot template match on Telecaster

├── test_images/                       # Evaluation/test set

├── vertical_lines_tool.py             # (Unused) vertical line checker/debugger
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```


| Script/Filename                         | Purpose                                                     |
| --------------------------------------- | ----------------------------------------------------------- |
| `1_exploratory.py`                      | First-pass EDA on image masks and layout                    |
| `2_template_creation_new.py`            | Early draft of body/head template creation                  |
| `3_template_matching.py`                | First implementation of body/head matching                  |
| `3.2_*.py` and `3.5_*.py`               | Experimental naive matching with rotation & scaling         |
| `4_guitar_part_position_helper.py`      | Calculates body/head bounding boxes                         |
| `5_new_template_idea_w_orientations.py` | Considers template rotation/orientation for better matching |
| `6_new_template_creation.py`            | Final body/head template creation                           |
| `7_new_template_matching.py`            | Matches templates to test images                            |
| `8_Final_Code.py`                       | 🔥 Run this to reproduce all final results                  |


### Reproducing Results
1. Install dependencies -> pip install -r requirements.txt
2. Run the final pipeline -> python template_matching_code/template_matching_code/8_Final_Code.py


### Notes
- Data is not augmented or normalized—relies on consistent backgrounds.
- Template matching is custom, no OpenCV keypoint/feature detectors are used in final version.
- Trial keypoint scripts are non-functional.
