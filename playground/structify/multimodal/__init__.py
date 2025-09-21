"""
Multimodal Intelligence Demos for OmniAdapters.

This module provides demonstrations of image and audio analysis capabilities
using structured outputs across multiple LLM providers.
"""

from playground.structify.multimodal.models import (
    ActionItem,
    AnalysisConfidence,
    AudioAnalysis,
    AudioSegment,
    BoundingBox,
    ChartData,
    DetectedObject,
    ExtractedText,
    ImageAnalysis,
    KeyPoint,
    MultimodalAnalysis,
    SceneUnderstanding,
    SentimentScore,
    SpeakerSegment,
    TopicClassification,
    TranscriptData,
)
from playground.structify.multimodal.utils import (
    compare_providers,
    create_audio_analysis_table,
    create_image_analysis_table,
    create_srt_subtitles,
    display_analysis_results,
    export_to_markdown,
    load_audio,
    load_image,
)

__all__ = [
    "ImageAnalysis",
    "AudioAnalysis",
    "MultimodalAnalysis",
    "DetectedObject",
    "BoundingBox",
    "ExtractedText",
    "ChartData",
    "SceneUnderstanding",
    "TranscriptData",
    "AudioSegment",
    "SpeakerSegment",
    "SentimentScore",
    "KeyPoint",
    "ActionItem",
    "TopicClassification",
    "AnalysisConfidence",
    "load_image",
    "load_audio",
    "display_analysis_results",
    "create_image_analysis_table",
    "create_audio_analysis_table",
    "export_to_markdown",
    "create_srt_subtitles",
    "compare_providers",
]
