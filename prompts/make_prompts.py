"""Build the T1 prompt set: 6 objects x 2 scenes x {left, right} = 24 prompts.

Phrasing ("light" vs "shadow") is assigned by parity of (object, scene, direction)
so that every direction, every scene, and every object gets an even split of
phrasings, and phrasing is never confounded with direction.

Direction convention used everywhere in this repo:
  target_dir = side the LIGHT COMES FROM, in image coordinates ("left"/"right").
  A "shadow" phrasing therefore names the opposite side (light from left -> shadow
  falls to the right).

Usage: python make_prompts.py  -> writes t1_prompts.jsonl next to this file.
"""
import json
from pathlib import Path

OBJECTS = ["a red ball", "a wooden chair", "a fire hydrant",
           "a traffic cone", "a park bench", "a potted plant"]
SCENES = ["an open grass field", "an empty parking lot"]
DIRS = ["left", "right"]
OPP = {"left": "right", "right": "left"}

TEMPLATES = {
    # names where the light comes from
    "light": ("A photo of {obj} standing alone in {scene} on a clear sunny day, "
              "strong sunlight coming from the {light} side of the image, "
              "casting a long hard shadow on the ground."),
    # names where the shadow falls (the opposite side)
    "shadow": ("A photo of {obj} standing alone in {scene} on a clear sunny day, "
               "strong sunlight, its long hard shadow falling to the {shadow} side "
               "of the image."),
}


def main():
    rows = []
    for oi, obj in enumerate(OBJECTS):
        for si, scene in enumerate(SCENES):
            for di, d in enumerate(DIRS):
                phrasing = "light" if (oi + si + di) % 2 == 0 else "shadow"
                prompt = TEMPLATES[phrasing].format(obj=obj, scene=scene,
                                                    light=d, shadow=OPP[d])
                rows.append(dict(prompt_id=f"p{len(rows):02d}", object=obj, scene=scene,
                                 target_dir=d, phrasing=phrasing, prompt=prompt))
    out = Path(__file__).with_name("t1_prompts.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # balance check
    from collections import Counter
    print(len(rows), "prompts ->", out)
    print("dir x phrasing:", Counter((r["target_dir"], r["phrasing"]) for r in rows))
    print("obj x phrasing:", Counter((r["object"], r["phrasing"]) for r in rows))


if __name__ == "__main__":
    main()
