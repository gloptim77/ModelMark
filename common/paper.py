from __future__ import annotations

import json
import platform
import subprocess
from html import escape
from pathlib import Path
from pathlib import Path
from typing import Iterable
from playwright.sync_api import sync_playwright

import torch
import pandas as pd

import config
import logging
logger = logging.getLogger(__name__)

class Paper:

	"""
	This class forms a paper-style table based on model evaluation data.
	And saves results in both .html and .png formats that are easy to embed to any research paper or a website.
	"""

	def __init__(self):

		self.html_path = "result.html"
		self.png_path = "result.png"
		self.machine_info = self._get_machine_info()

	def _extract_mean(self, value) -> float:
		"""Extract the mean from values such as '0.3561 ± 0.0275'."""
		if isinstance(value, str):
			return float(value.split("±")[0].strip())
		return float(value)
	
	def _highlight_best(
		self,
		row: "pd.Series",
		columns: Iterable[tuple],
		higher_is_better: set[str] | bool = False,
	) -> set[tuple]:
		"""
		Given one row (a Series indexed by (model, metric) tuples) and the
		(model, metric) column tuples that actually exist for this row,
		return the set of (model, metric) tuples holding the best value
		for each metric.
	
		higher_is_better:
			- False (default): every metric is "lower is better"
			- True: every metric is "higher is better"
			- a set of metric names: only those metrics are treated as
			  "higher is better"; everything else is "lower is better"
	
		Only compares columns that actually exist for this row, so models
		with different metric sets never cause a KeyError.
		"""
		best: set[tuple] = set()
	
		by_metric: dict[str, list[tuple]] = {}
		for model, metric in columns:
			by_metric.setdefault(metric, []).append((model, metric))
	
		for metric, cols in by_metric.items():
			values = {}
			for col in cols:
				try:
					values[col] = self._extract_mean(row[col])
				except (ValueError, TypeError, KeyError):
					continue
				
			if not values:
				continue
			
			want_max = higher_is_better is True or (
				isinstance(higher_is_better, set) and metric in higher_is_better
			)
			best_col = max(values, key=values.get) if want_max else min(values, key=values.get)
			best.add(best_col)
	
		return best
	
	def _ensure_contiguous_datasets(self, df: "pd.DataFrame") -> "pd.DataFrame":
		"""
		Reorder rows so every row belonging to the same top-level index
		value ("Dataset") is contiguous, preserving first-appearance order
		of datasets and original order within each dataset.
	
		Required for HTML rowspan on the dataset cell to render correctly
		— a rowspan just covers the next N rows in document order, so if
		a dataset's rows aren't contiguous, the span will visually cover
		rows belonging to a different dataset.
		"""
		dataset_order = list(dict.fromkeys(df.index.get_level_values(0)))
		if len(dataset_order) <= 1:
			return df
		return pd.concat([df.xs(d, level=0, drop_level=False) for d in dataset_order])
	
	def _build_stats_rows(
		self,
		model_stats: dict,
		models: list,
		metrics_by_model: dict,
		highlight_best: bool,
	) -> str:
		"""
		Build one summary row per (label -> per-model value) entry in
		model_stats. Unlike the main body, these are per-model, not
		per-metric — e.g. training time, parameter count, GFLOPs — so
		each value cell spans all of that model's metric columns.
	
		Each entry in model_stats is either:
			- a list/array of values, one per model, in the same order
			  as `models` (i.e. the order models appear left-to-right
			  in the table), or
			- a dict mapping model name -> value.
	
		highlight_best bolds the smallest value per row (time, params,
		and FLOPs are always "smaller is better").
		"""
		rows_html = []
	
		for label, values in model_stats.items():
			if isinstance(values, dict):
				by_model = values
			else:
				values = list(values)
				if len(values) != len(models):
					raise ValueError(
						f"model_stats[{label!r}] has {len(values)} values but there are "
						f"{len(models)} models ({models}); pass one value per model, in "
						f"that order, or a dict keyed by model name."
					)
				by_model = dict(zip(models, values))
	
			numeric = {}
			for model in models:
				try:
					numeric[model] = self._extract_mean(by_model[model])
				except (ValueError, TypeError, KeyError):
					continue
				
			best_models = set()
			if highlight_best and numeric:
				best_val = min(numeric.values())
				best_models = {m for m, v in numeric.items() if v == best_val}
	
			cells = [f'<th class="stats-label" colspan="2">{escape(str(label))}</th>']
			for model in models:
				colspan = len(metrics_by_model[model])
				classes = ["stats-value"]
				if model in best_models:
					classes.append("best")
				value = by_model.get(model, "")
				cells.append(
					f'<td class="{" ".join(classes)}" colspan="{colspan}">{escape(str(value))}</td>'
				)
	
			rows_html.append('<tr class="stats-row">' + "".join(cells) + "</tr>")
	
		return "".join(rows_html)
	
	def dataframe_to_html(
		self,
		df: "pd.DataFrame",
		model_stats: dict,
		metadata: dict,
		title: str = "Neural Network Model Evaluation",
		subtitle: str | None = None,
		higher_is_better: set[str] | bool = False,
		zebra_stripe: bool = True,
		highlight_best_stats: bool = True,
	) -> str:
		"""
		Create a complete standalone HTML report from a DataFrame.
	
		Parameters
		----------
		df:
			The multi-index dataframe with 2 columns and rows levels, 
			should be formed as in the example.
		model_stats:
			Per-model summary rows appended below the results,
			e.g. training time / parameter count / GFLOPs — one value per
			model rather than per metric, so each value spans that
			model's whole column group. Example:
	
				model_stats={
					"Time (s)": [12.3, 9.8],
					"Params (M)": [25.6, 22.1],
					"GFLOPs": [4.1, 3.9],
				}
	
			Each list must be in the same order as the models appear in
			the table (left to right); a dict of {model_name: value} is
			also accepted per row if you'd rather not rely on order.
		metadata:
			The dictonary with system and hardware info.
		title:
			The name of the table or research.
		subtitle:
			Optional additional info about the research-related details.
		html_path:
			Path to the file that should contain the generated html code. 
		higher_is_better:
			Passed through to the best-value highlighting. False (default)
			treats every metric as lower-is-better; pass True to flip all
			metrics, or a set of metric names to flip only those (e.g.
			{"Accuracy", "F1"}).
		zebra_stripe:
			Light alternating row shading for on-screen readability.
			Set False for a stricter print/booktabs look.
		highlight_best_stats:
			Bold the smallest value in each model_stats row (time,
			params, and FLOPs are always "smaller is better").
	
		The function writes the HTML file and returns the generated HTML.
		"""

		html_path = self.html_path
		html_path = Path(html_path)
	
		# ---------------------------------------------------------
		# Validate DataFrame structure
		# ---------------------------------------------------------
	
		if not isinstance(df.index, pd.MultiIndex):
			raise ValueError("DataFrame index must be a MultiIndex, e.g. ['Dataset', 'Context'].")
	
		if not isinstance(df.columns, pd.MultiIndex):
			raise ValueError("DataFrame columns must be a MultiIndex, e.g. ['Model', 'Metric'].")
	
		if df.index.nlevels != 2:
			raise ValueError(f"Expected 2 index levels, got {df.index.nlevels}.")
	
		if df.columns.nlevels != 2:
			raise ValueError(f"Expected 2 column levels, got {df.columns.nlevels}.")
	
		df = self._ensure_contiguous_datasets(df)
	
		# Names used in the table (display labels only — lookups below
		# are positional, so these can be anything, including None).
		column_name_1 = df.columns.names[0] or "Model"
		column_name_2 = df.columns.names[1] or "Metric"
	
		# ---------------------------------------------------------
		# Header
		# ---------------------------------------------------------
	
		model_values = list(df.columns.get_level_values(0))
		models = list(dict.fromkeys(model_values))  # preserve order
	
		metrics_by_model = {}
		for model in models:
			metrics_by_model[model] = list(
				dict.fromkeys(metric for current_model, metric in df.columns if current_model == model)
			)
	
		# First header row: model names, each with a short "cmidrule"
		# underline spanning only its own metric columns.
		header_row_1 = f"""
			<tr class="header-main">
				<th class="index-header corner-label" colspan="2">{escape(str(column_name_1))}</th>
		"""
		for model in models:
			colspan = len(metrics_by_model[model])
			header_row_1 += f"""
				<th colspan="{colspan}" class="model-header">
					<div class="model-header-inner">{escape(str(model))}</div>
				</th>
			"""
		header_row_1 += "</tr>"
	
		# Second header row: metric names, with a full-width rule beneath
		# separating the header from the body.
		header_row_2 = f"""
			<tr class="header-metric">
				<th class="corner-placeholder corner-label" colspan="2">{escape(str(column_name_2))}</th>
		"""
		for model in models:
			for metric in metrics_by_model[model]:
				header_row_2 += f"""
					<th>{escape(str(metric))}</th>
				"""
		header_row_2 += "</tr>"
	
		# ---------------------------------------------------------
		# Body
		# ---------------------------------------------------------
	
		body_rows = []
	
		dataset_rowspans = {}
		for dataset, group in df.groupby(level=0, sort=False):
			dataset_rowspans[dataset] = len(group)
	
		dataset_rendered = set()
		previous_dataset = None
	
		for row_number, (index, row) in enumerate(df.iterrows()):
			dataset = index[0]
			context = index[1]
	
			best_columns = self._highlight_best(row, df.columns, higher_is_better=higher_is_better)
	
			cells = []
	
			# Dataset cell — only on the first row of each dataset block,
			# spanning all of that dataset's rows.
			if dataset not in dataset_rendered:
				rowspan = dataset_rowspans[dataset]
				cells.append(
					f'<th class="dataset-cell" rowspan="{rowspan}">{escape(str(dataset))}</th>'
				)
				dataset_rendered.add(dataset)
	
			# Context cell
			cells.append(f'<th class="context-cell">{escape(str(context))}</th>')
	
			# Metric cells
			for model in models:
				for metric in metrics_by_model[model]:
					value = row[(model, metric)]
					classes = ["value-cell"]
					if (model, metric) in best_columns:
						classes.append("best")
					cells.append(f'<td class="{" ".join(classes)}">{escape(str(value))}</td>')
	
			row_classes = []
			if zebra_stripe and row_number % 2 == 1:
				row_classes.append("striped")
			if previous_dataset is not None and dataset != previous_dataset:
				row_classes.append("dataset-start")
			previous_dataset = dataset
	
			body_rows.append(f'<tr class="{" ".join(row_classes)}">' + "".join(cells) + "</tr>")
	
		# ---------------------------------------------------------
		# Per-model summary rows (Time, Params, GFLOPs, ...)
		# ---------------------------------------------------------
	
		stats_rows_html = ""
		if model_stats:
			stats_rows_html = self._build_stats_rows(model_stats, models, metrics_by_model, highlight_best_stats)
	
		# ---------------------------------------------------------
		# Metadata
		# ---------------------------------------------------------
	
		metadata_rows = []
		for key, value in metadata.items():
			metadata_rows.append(
				f"""
				<div class="metadata-row">
					<div class="metadata-label">{escape(str(key))}</div>
					<div class="metadata-value">{escape(str(value))}</div>
				</div>
				"""
			)
		metadata_html = "\n".join(metadata_rows)
	
		subtitle_html = ""
		if subtitle:
			subtitle_html = f'<div class="subtitle">{escape(subtitle)}</div>'
	
		# ---------------------------------------------------------
		# Complete document
		# ---------------------------------------------------------
	
		html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{escape(title)}</title>
<style>

* {{
	box-sizing: border-box;
}}

body {{
	margin: 0;
	padding: 0;
	background: #ffffff;
	color: #111111;
	font-family: "Times New Roman", Georgia, serif;
}}

.report {{
	display: inline-block;
	padding: 20px;
}}

.title {{
	font-size: 22px;
	font-weight: 600;
	margin-bottom: 4px;
}}

.subtitle {{
	color: #666666;
	font-size: 13px;
	margin-bottom: 22px;
	overflow-wrap: break-word;
	white-space: pre-wrap;
}}

/* ============================================================
   TABLE — booktabs style: no vertical rules, rules top/bottom
   ============================================================ */

table {{
	border-collapse: collapse;
	font-size: 13.5px;
	border-top: 2px solid #000000;
	border-bottom: 2px solid #000000;
}}

thead th {{
	padding: 6px 16px;
	text-align: center;
	font-weight: 400;
}}

.header-main th {{
	font-weight: 600;
	padding-bottom: 2px;
}}

.header-main .model-header {{
	font-size: 14px;
}}

.model-header-inner {{
	display: block;
	margin: 0 6px 4px;
	padding-bottom: 4px;
	border-bottom: 1px solid #000000;
}}

.header-metric th {{
	font-size: 12.5px;
	font-weight: 400;
	color: #333333;
	border-bottom: 1px solid #000000;
	padding-bottom: 6px;
}}

.corner-label {{
	font-size: 12.5px;
	font-weight: 400;
	color: #000000;
	text-transform: uppercase;
	letter-spacing: 0.03em;
}}

/* ============================================================
   BODY
   ============================================================ */

tbody th,
tbody td {{
	padding: 6px 16px;
}}

.dataset-cell {{
	text-align: left;
	vertical-align: middle;
	font-weight: 600;
	white-space: nowrap;
}}

.context-cell {{
	text-align: left;
	padding-left: 24px !important;
	vertical-align: middle;
	white-space: nowrap;
	font-style: italic;
	color: #444444;
	font-weight: 400;
}}

.value-cell {{
	text-align: center;
	white-space: nowrap;
	font-variant-numeric: tabular-nums;
}}

tr.striped td,
tr.striped th {{
	background: #f7f7f7;
}}

tr.dataset-start td,
tr.dataset-start th {{
	border-top: 1px solid #cccccc;
}}

/* ============================================================
   PER-MODEL SUMMARY STATS (time, params, FLOPs, ...)
   ============================================================ */

tbody.stats tr:first-child td,
tbody.stats tr:first-child th {{
	border-top: 1px solid #000000;
}}

.stats-label {{
	text-align: left;
	font-size: 12px;
	font-style: italic;
	font-weight: 400;
	color: #555555;
}}

.stats-value {{
	text-align: center;
	white-space: nowrap;
	font-variant-numeric: tabular-nums;
	font-size: 12.5px;
	color: #333333;
}}

/* ============================================================
   BEST RESULTS
   ============================================================ */

.best {{
	font-weight: 700;
}}

/* ============================================================
   METADATA
   ============================================================ */

.metadata {{
	margin-top: 20px;
	padding-top: 10px;
	font-size: 12.5px;
	color: #444444;
}}

.metadata-title {{
	margin-bottom: 8px;
	color: #111111;
	font-size: 16px;
	font-weight: 600;
}}

.metadata-row {{
	display: grid;
	grid-template-columns: 110px auto;
	margin: 3px 0;
}}

.metadata-label {{
	font-weight: 600;
	color: #222222;
}}

.metadata-value {{
	color: #000000;
}}

</style>
</head>
<body>

<div class="report">

	<div class="title">{escape(title)}</div>
	{subtitle_html}

	<table>
		<thead>
			{header_row_1}
			{header_row_2}
		</thead>
		<tbody>
			{"".join(body_rows)}
		</tbody>
		{f'<tbody class="stats">{stats_rows_html}</tbody>' if stats_rows_html else ""}
	</table>

	<div class="metadata">
		<div class="metadata-title">Experimental setup</div>
		{metadata_html}
	</div>

</div>

</body>
</html>
	"""
	
		html_path.parent.mkdir(parents=True, exist_ok=True)
		html_path.write_text(html, encoding="utf-8")
	
		return html
	
	def html_to_png(self, scale: int = 2,):
		"""Convert the .html file to the .png image."""

		html_path = Path(self.html_path).resolve()
		png_path = Path(self.png_path)
	
		png_path.parent.mkdir(
			parents=True,
			exist_ok=True,
		)
	
		with sync_playwright() as p:
			browser = p.chromium.launch()
	
			page = browser.new_page(
				device_scale_factor=scale,
			)
	
			page.goto(
				html_path.as_uri(),
				wait_until="networkidle",
			)
	
			# Find the actual report element
			report = page.locator(".report")
	
			# Take a screenshot of ONLY that element
			report.screenshot(
				path=str(png_path),
			)
	
			browser.close()

	def _get_cpu_name(self) -> str:

		system = platform.system()

		if system == "Windows":
			# Uses Windows Management Instrumentation (WMI) via command line
			try:
				command = "wmic cpu get name"
				output = subprocess.check_output(command, shell=True).decode().strip()
				# The output contains a header 'Name', split and grab the actual value
				return output.split("\n")[1].strip()
			except Exception:
				return platform.processor()

		elif system == "Darwin":  # macOS
			# Queries the sysctl kernel utility for the CPU brand string
			try:
				command = ["sysctl", "-n", "machdep.cpu.brand_string"]
				return subprocess.check_output(command).decode().strip()
			except Exception:
				return platform.processor()

		elif system == "Linux":
			# Reads the system's virtual /proc/cpuinfo file directly
			try:
				with open("/proc/cpuinfo", "r") as f:
					for line in f:
						if "model name" in line:
							# Extract the name after the colon
							return line.split(":", 1)[1].strip()
			except Exception:
				# Fallback if /proc/cpuinfo isn't accessible (e.g., some containers)
				try:
					output = subprocess.check_output("lscpu", shell=True).decode()
					for line in output.splitlines():
						if "Model name:" in line:
							return line.split(":", 1)[1].strip()
				except Exception:
					return platform.processor()

		return "Unknown Processor"

	def _get_machine_info(self) -> dict:
		"""Get the info about system and hardware, returns as dictonary."""

		# OS
		os_name = f"{platform.system()} {platform.release()}"
		os_version = f"{platform.version()}"
		arch = f"{platform.machine()}"
		# CPU
		cpu_name = f"{self._get_cpu_name()}"	
		# GPU
		gpu_name = "_"
		if torch.cuda.is_available():
			gpu_name = torch.cuda.get_device_name(0)
		# Python version
		python_version = f"{platform.python_version()}"
		# Pytorch version
		pytorch_version = f"{torch.__version__}"
	
		return {
			"OS": os_name + " " + os_version + " " + arch,
			"Python": python_version,
			"PyTorch": pytorch_version,
			"CPU": cpu_name,
			"GPU": gpu_name
		}

	def report(self, df, title, subtitle, model_stats):

		# Print the report to the console #

		# Title
		print("=" * 128)
		print(" " * 60 + "Report:")
		print("=" * 128)
		# DataFrame
		print("-" * 128)
		print(df)
		print("-" * 128)
		# Stats
		print("Training stats:")
		report_model_names = list(config.data_config)
		print(f"{'Metric':<18} | " + " | ".join(f"{h:>10}" for h in report_model_names))
		print("-" * 128)
		for key, values in model_stats.items():
			print(f"{key:<18} | " + " | ".join(f"{float(v):>10.2f}" for v in values))
		print("-" * 128)
		# Meta
		print("Experimental setup:")
		print(json.dumps(self.machine_info, indent = 4))
		print("=" * 128)

		# Convert and save the report #

		self.dataframe_to_html(df=df, model_stats=model_stats, metadata=self.machine_info, title=title, subtitle=subtitle)
		self.html_to_png()
