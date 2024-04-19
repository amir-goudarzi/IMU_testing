# Self-Supervised Pre-training for Inertial Movement Unit (IMU)

## 1. Data cleaning

We are using EPIC-Kitchens dataset which has GoPro IMU data. The average source sampling rates according to [GoPro Specs](https://github.com/gopro/gpmf-parser/tree/main) follows:
|||
| ------------- | -------|
| Accelerometer | 200 Hz |
| Gyroscope     | 400 Hz |

Preprocessing steps:
1. Load files (accl, gyro) from a video, then interpolate them
2. Store new data in a EPIC-Kitchens directory format