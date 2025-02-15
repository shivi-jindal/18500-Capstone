# 18500-Capstone: Write on Cue
## Shivi Jindal, Grace Li, Deeya Patel

Welcome to the codebase for our capstone project, called Write on Cue. Our capstone project aims to create a music composition system that transcribes flute performances into sheet music. This will address the challenge faced by musicians who want to capture their improvisations, rehearsals, or live performances without manually transcribing the music. The project scope includes a pipeline to process audio signals from a flute based on an inputted BPM, extracting the correct pitch and rhythm (which will be referred to as note) being played, and outputting a digital sheet music file. The implementation will focus on analyzing the frequency and pitch of the flute audio to ensure that there is a high transcription accuracy. We will wrap this pipeline into a basic web or mobile app that provides an interface for the user to play a piece and view the generated score. As a stretch goal, we hope to incorporate a generative AI feature that will suggest potential next notes to assist the composer in developing their composition.

### Environment Setup

```git clone https://github.com/shivi-jindal/18500-Capstone```

Install dependencies: 
```conda env create -f capstone_env.yml```
```conda activate capstone```

To deactivate the environment:
```conda deactivate```

To add any changes:
```git add .
git commit -m "commit message"
git push
```

All audio recordings should be added to the "Audio" directory and signal processing algos/code in "Signal Processing" directory.