"""Human-readable measurements accompanying the full saved-history JSON."""
from __future__ import annotations

import json
from pathlib import Path


def number(value, digits=1):
    return 'unavailable' if value is None else f'{value:,.{digits}f}'


def epoch_label(value):
    """Preserve fractional saved epochs instead of rounding display labels."""
    return str(value).removesuffix('.0')


def label(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def markdown(report):
    provenance = report['provenance']
    lines = ['# Basin and mountain history review', '',
             f"Run: `{provenance['run_id']}`. Selected exact saved epochs: "
             + ', '.join(epoch_label(t) for t in provenance['selected_epochs_myr']) + ' Myr.', '',
             'This review samples fixed geographic paths and separately follows saved material markers. '
             'A mountain moving out of a fixed path is not evidence of erosion. '
             'Samples do not establish global extrema or exact shorelines.', '',
             '## Mountain profiles', '',
             'Widths below are along each selected path. Highland means elevation at least 2 km by default; '
             'the summit cap is the contiguous portion within 1 km of the sampled summit. '
             'Neither metric establishes a two-dimensional plateau. A censored width reaches a path endpoint.', '']
    for profile in report['mountains'].get('profiles', []):
        lines += [f"### {label(profile['name'])}", '',
                  '| Epoch (Myr) | Sampled summit (m) | Relief (m) | Highland width (km) | Summit cap (km) | Left/right low (m) | Censored? |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for epoch in profile['epochs']:
            m = epoch['metrics']
            censored = any(m.get(k) for k in ['highland_censored_left', 'highland_censored_right',
                                             'cap_censored_left', 'cap_censored_right'])
            row = [epoch_label(epoch['time_myr']), number(m.get('sampled_peak_m')),
                   number(m.get('sampled_relief_m')), number(m.get('highland_width_km')),
                   number(m.get('summit_cap_width_km')),
                   number(m.get('left_adjacent_low_m'))+' / '+number(m.get('right_adjacent_low_m')),
                   'yes' if censored else 'no']
            lines.append('| '+' | '.join(row)+' |')
        lines += ['', 'The JSON retains every sampled distance/elevation and the available selected-sheet, '
                  'stack-support and thermal components for checking the height calculation.', '']
    lines += ['## Ocean gaps and boundary crossings', '',
              'These are intersections with selected paths, not named geological basin identities. '
              'Gap numbers restart at each epoch. Flooded continental crust is distinct from oceanic crust. '
              'Widths and age summaries are sampled; transition brackets quantify sampling uncertainty.', '']
    for epoch in report['basins'].get('epochs', []):
        lines += [f"### {epoch_label(epoch['time_myr'])} Myr", '',
                  '| Path | Oceanic gap | Width ± sampling uncertainty (km) | Ocean age min / mean / max (Myr) | Path-truncated? |',
                  '| --- | --- | --- | --- | --- |']
        for path in epoch['transects']:
            gaps = path.get('submerged_oceanic_crust_intervals')
            if gaps is None:
                lines.append(f"| {label(path['id'])} | unavailable | unavailable | unavailable | unavailable |")
            elif not gaps:
                lines.append(f"| {label(path['id'])} | none sampled | — | — | — |")
            else:
                for i, gap in enumerate(gaps):
                    ages = ' / '.join(number(gap.get('ocean_age_'+stat+'_myr')) for stat in ['min', 'mean', 'max'])
                    truncated = gap.get('truncated_at_start') or gap.get('truncated_at_end')
                    lines.append('| '+' | '.join([label(path['id']), str(i+1),
                        number(gap.get('sampled_width_km'))+' ± '+number(gap.get('width_uncertainty_km')),
                        ages, 'yes' if truncated else 'no'])+' |')
        lines += ['', 'Boundary intersections along those paths:', '']
        for path in epoch['transects']:
            boundaries = path['boundaries']
            crossings = boundaries.get('intersections')
            description = ('unavailable' if crossings is None else
                ', '.join(f"{row['type']} at {number(row['distance_km'])} km" for row in crossings)
                or 'none recorded')
            if boundaries.get('collinear_overlaps'):
                description += '; collinear boundary overlap retained separately in JSON'
            lines.append(f"- **{label(path['id'])}:** {description}.")
        lines.append('')
    lines += ['## Material-marker observations', '',
              'Markers are representative material samples. A marker height need not equal the displayed '
              'upper surface where sheets overlap. Activity labels are observations from the selected '
              'intervals; a shutdown time between those intervals is not inferred.', '']
    for history in report['mountains'].get('material_histories', []):
        lines += [f"### Marker {history['trace_id']}", '',
                  '| Epoch (Myr) | Present | Longitude / latitude | Representative relief (m) | Observed activity |',
                  '| --- | --- | --- | --- | --- |']
        for epoch in history['epochs']:
            lines.append('| '+' | '.join([epoch_label(epoch['time_myr']), str(epoch.get('present', False)),
                number(epoch.get('longitude_deg'))+' / '+number(epoch.get('latitude_deg')),
                number(epoch.get('representative_relief_m')), label(epoch.get('activity', 'unavailable'))])+' |')
        lines.append('')
    lines += ['## Provenance and limits', '',
              'The full `review.json` includes exact input hashes, analysis-helper hashes, paths, '
              'selection rules, marker-selection distances, per-sample values and source-specific limitations. '
              'It can be regenerated from the same preserved snapshots. Input files and the observed '
              'committed prefix were checked again after analysis. New frames may append safely.', '',
              'No simulation step, procedural fine terrain, erosion multiplier or physical repair was applied. '
              'Unavailable quantities remain unavailable. Cumulative global ocean totals are not allocated '
              'to an individual path or claimed as its local consumption.', '']
    return '\n'.join(lines)


def write_markdown(report, output):
    target = Path(output)/'REVIEW.md'
    target.write_text(markdown(report), encoding='utf-8')
    return target


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review_json', type=Path)
    args = parser.parse_args()
    write_markdown(json.loads(args.review_json.read_text(encoding='utf-8')), args.review_json.parent)
