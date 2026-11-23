Engineering camera-read optimization only. Preserve the three original policy RGB
views, commands, actual reads, poses/lights/RNG, physics, source/phase/query budgets,
weights and all original controls. A marked policy task may temporarily omit only
the unused front_camera from the original static capture/read lists. Do not remove
camera objects or change image sizes, shaders, dtype, lighting or wrist updates.
Restore the exact original list objects even on exceptions. No video/extra modality
use; unsupported modes refuse. Original reference/expert tasks remain untouched.
CPU checks cannot establish native image parity; one bounded200s first32/nextRGB
GPU parity probe first, actual all time charged to currentVAL18000/carry12711.033.
Account3/CPU16/24563/34+35/36/strictNVML/Rule4 remain; no filler or failed retry.
