# Eval recording script (eval_02)

A script to read aloud for a board-mic evaluation recording: about 4½ minutes, with 10 each of `forward`,
`backward` and `stop`.

Every keyword is followed by at least 3 s of silence or a non-keyword sentence before the next one. That's
longer than `score_results.py`'s 1.5 s suppression window and its ~1.5 s matching tolerance, so two
keywords never compete for the same detection.

## Before you start
- Record with
  `python receive_and_play.py -p <port> -o eval_recordings/eval_02_script.wav`
  (use the next free number for later recordings).
- Keep the board in one place and move yourself to the distances given.
- Speak naturally. The filler sentences don't need to be word for word; just keep about 3 s between keywords.
- *(pause)* means stay silent for about 3 s (count to three).

---

## A. Room tone: 30 cm from the board, silent for 30 s

## B. Isolated keywords: 30 cm, normal voice
*(pause)* after each one:

**forward** · **stop** · **backward** · **stop** · **forward** · **backward**

## C. Keywords in sentences: 30 cm
Read straight through:
1. Okay, let's see if this thing can hear me properly today.
2. Go **forward** a little bit, please.
3. The weather has been pretty grey all week, hasn't it.
4. Now **stop** right there.
5. I was thinking about what to have for dinner later on.
6. Move **backward** slowly.
7. There were four people waiting outside the shop this morning.
8. Keep going **forward** until you reach the wall.
9. My phone battery keeps dying faster than it used to.
10. Please **stop** the robot now.
11. I need to remember to call my sister at the weekend.
12. Can you reverse it and go **backward** again?

## D. Similar-sounding words, no keywords: 30 cm
These test for false accepts.
1. Follow the path towards the top of the hill.
2. Four or five of them came back for more.
3. We stepped onto the shop floor and looked toward the front.
4. The forecast for tomorrow looks a bit better.
5. Put the backpack on top of the desk.
6. He moved onward to the fourth chapter.
7. *(Optional: label these two `ignore`.)* The bus stopped suddenly. He walked backwards into the door.

## E. About 1 m away, normal voice
- **stop** *(pause)* **forward** *(pause)* **backward** *(pause)*
- Alright, from over here, go **forward** two steps.
- I'm standing a bit further away from the board now.
- And **stop**.
- Let's try one more from this distance.
- Take it **backward** a little.

## F. About 2 m away, slightly louder
- **backward** *(pause)* **stop** *(pause)* **forward** *(pause)*
- From across the room, **forward**, please.
- This is the far side of the room now.
- Okay, **stop** there.
- One more test at this distance.
- Go **backward**.

## G. Back at 30 cm, varying the delivery
- Fast: "Go **forward**, go on, go on, keep going."
- *(pause)* Quietly: **backward**
- *(pause)* Slow and drawn out: **stooop**

## H. TV or music on at a moderate volume, you about 1 m from the board
- Let's see if it works with the TV on.
- Go **forward**.
- There's quite a lot of noise in the room now.
- **Stop**.
- I wonder if it can pick me out from all this.
- **Backward**, please.
- Leave the TV on for 20 s without speaking, then turn it off.

## I. Silent for 10 s, then Ctrl-C

---

**Tally:** 10 each of `forward`, `backward` and `stop`, plus about 20 non-keyword sentences, 60 s of silence
and 40 s of noise.

## Labelling notes
- Label the 30 keywords, and mark the two optional variants in D as `ignore`.
- If the TV happens to say a keyword, label that as `ignore` too.
- The section letters help you find your place in Audacity.
