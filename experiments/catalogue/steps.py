"""Stable module names for catalogue pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineModules:
    prepare_dataset: str = "experiments.catalogue.commands.prepare_dataset"
    extract_evidence: str = "experiments.catalogue.commands.extract_evidence"
    build_inputs: str = "experiments.catalogue.commands.build_inputs"
    build_rough_atoms: str = "experiments.catalogue.commands.build_rough_atoms"
    qualify_atoms: str = "experiments.catalogue.commands.qualify_atoms"
    prebasket_atoms: str = "experiments.catalogue.commands.prebasket_atoms"
    rebasket_atoms: str = "experiments.catalogue.commands.rebasket_atoms"
    merge_baskets: str = "experiments.catalogue.commands.merge_baskets"
    collapse_other_baskets: str = "experiments.catalogue.commands.collapse_other_baskets"
    summarize_rebasket: str = "experiments.catalogue.commands.summarize_rebasket"
    find_pending_phrases: str = "experiments.catalogue.commands.find_pending_phrases"
    review_pending_phrases: str = "experiments.catalogue.commands.review_pending_phrases"
    build_node_mapping: str = "experiments.catalogue.commands.build_node_mapping"
    group_aspects: str = "experiments.catalogue.commands.group_aspects"
    check_aspect_boundaries: str = "experiments.catalogue.commands.check_aspect_boundaries"
    score_aspect_tree: str = "experiments.catalogue.commands.score_aspect_tree"
    split_atoms: str = "experiments.catalogue.commands.split_atoms"
    build_dashboard_data: str = "experiments.catalogue.commands.build_dashboard_data"
    build_dashboard_aspect_tree: str = "experiments.catalogue.commands.build_dashboard_aspect_tree"
    build_flat_subclusters: str = "experiments.catalogue.commands.build_flat_subclusters"
    build_statistical_subclusters: str = "experiments.catalogue.commands.build_statistical_subclusters"
    build_document_phrases: str = "experiments.catalogue.commands.build_document_phrases"
    extract_facets: str = "experiments.catalogue.commands.extract_facets"
    build_facets: str = "experiments.catalogue.commands.build_facets"
    name_atoms: str = "experiments.catalogue.commands.name_atoms"
    apply_dashboard_titles: str = "experiments.catalogue.commands.apply_dashboard_titles"
    audit_aspect_siblings: str = "experiments.catalogue.commands.audit_aspect_siblings"
    name_aspects: str = "experiments.catalogue.commands.name_aspects"
    # Three-layer co-association aspect rebuild (driver under aspects.*, not a
    # commands.* step): atom -> mid-group -> top aspect, replaces the old flat
    # aspect grouping/naming. See docs/paper-alignment.md.
    rebuild_aspects: str = "experiments.catalogue.aspects.rebuild_aspects"


MODULES = PipelineModules()
