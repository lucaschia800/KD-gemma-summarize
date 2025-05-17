# Gemma-KD

Distilling Gemma-9B to a Gemma-2B model for the downstream task of summarization.
The goal and evaluation was to test both soft label KD and Generalized KD using on-policy learning to reduce training-inference distribution mismatch between
9B Gemma teacher and 2B Gemma student. 
