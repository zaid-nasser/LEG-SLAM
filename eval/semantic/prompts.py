from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PromptSet:
    name: str
    templates: Sequence[str]


# ----------------------------------------------------------------------
# ImageNet / CLIP templates
# ----------------------------------------------------------------------

FULL_IMAGENET_TEMPLATES = (
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
)


MASKCLIP_TEMPLATES = (
    "there is a {} in the scene.",
    "there is the {} in the scene.",
    "this is a {} in the scene.",
    "this is the {} in the scene.",
    "this is one {} in the scene.",
)


SUBSET_IMAGENET_TEMPLATES = (
    "itap of a {}.",
    "a bad photo of a {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
)


# ----------------------------------------------------------------------
# Prompt sets
# ----------------------------------------------------------------------

FULL = PromptSet(
    name="full",
    templates=FULL_IMAGENET_TEMPLATES,
)


FULL_MASKCLIP = PromptSet(
    name="full+maskclip",
    templates=(
        FULL_IMAGENET_TEMPLATES
        + MASKCLIP_TEMPLATES
    ),
)


SUBSET = PromptSet(
    name="subset",
    templates=SUBSET_IMAGENET_TEMPLATES,
)


SUBSET_MASKCLIP = PromptSet(
    name="subset+maskclip",
    templates=(
        SUBSET_IMAGENET_TEMPLATES
        + MASKCLIP_TEMPLATES
    ),
)


MASKCLIP = PromptSet(
    name="maskclip",
    templates=MASKCLIP_TEMPLATES,
)


SIMPLE = PromptSet(
    name="simple",
    templates=(
        "a photo of a {}.",
    ),
)


PLURAL = PromptSet(
    name="plural",
    templates=(
        "a photo of {}s.",
    ),
)


IDENTITY = PromptSet(
    name="identity",
    templates=(
        "{}",
    ),
)


CUSTOM = PromptSet(
    name="custom",
    templates=(
        SUBSET_IMAGENET_TEMPLATES
        + (
            "a photo of many {}.",
            "a photo of {}s.",
        )
    ),
)

OPENSCENE = PromptSet(
    name="openscene",
    templates=(
        "a {} in a scene.",
    ),
)

SIGLIP2_DEFAULT = PromptSet(
    name="siglip2_default",
    templates=(
        "this is a photo of {}.",
    ),
)

SPACEFORMER = PromptSet(
    name="spaceformer",
    templates=(
        "a {} in a scene.",
        "a photo of a {} in a scene.",
        "a {}.",
        "a photo of a {}.",
        "there is a {} in the scene.",
        "a 3d model of a {}.",
        "a {} in a room.",
    ),
)


PROMPT_SETS = {
    prompt_set.name: prompt_set
    for prompt_set in (
        FULL,
        FULL_MASKCLIP,
        SUBSET,
        SUBSET_MASKCLIP,
        MASKCLIP,
        SIMPLE,
        PLURAL,
        IDENTITY,
        CUSTOM,
        OPENSCENE,
        SIGLIP2_DEFAULT,
        SPACEFORMER,

    )
}


DEFAULT_PROMPT_SET = OPENSCENE


def normalize_class_name(
    name: str,
) -> str:
    return name.replace(
        "-",
        " ",
    )


def build_prompt_groups(
    class_names: Sequence[str],
    prompt_set: PromptSet = DEFAULT_PROMPT_SET,
) -> list[list[str]]:

    return [
        [
            template.format(
                normalize_class_name(name)
            )
            for template in prompt_set.templates
        ]
        for name in class_names
    ]