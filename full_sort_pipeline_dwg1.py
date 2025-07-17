import sys
import psutil  # Required for memory preloading
from qtpy.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit, 
                           QSlider, QComboBox, QDialog, QDialogButtonBox, QListWidget, 
                           QAbstractItemView, QCheckBox, QProgressBar, QScrollArea,
                           QSpinBox, QDoubleSpinBox, QLineEdit)

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QFont
import numpy as np
import glob
from pathlib import Path
import tdt
import spikeinterface.full as si
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import json
import shutil
import os
import re
import pickle
from scipy.stats import ttest_ind
import scipy.spatial.distance

class SortingWorker(QThread):
    finished = Signal(bool, str)
    log_message = Signal(str)
    progress = Signal(int)

    def __init__(self, recordings, raw_recordings, tdt_data_objects, file_paths, sorter_name, sorter_params, output_base_dir):
        super().__init__()
        self.recordings = recordings
        self.raw_recordings = raw_recordings
        self.tdt_data_objects = tdt_data_objects
        self.file_paths = file_paths
        self.sorter_name = sorter_name
        self.sorter_params = sorter_params  # Now passed from GUI
        self.output_base_dir = output_base_dir
        self.sorting_results = []
        self.unit_metrics = []


    def get_sorter_params(self, sorter_name):
        """Get appropriate parameters for different sorters"""
        params = {}

        if sorter_name == 'tridesclous2':
            params = {
                'freq_min': 300,
                'freq_max': 6000,
                'detect_threshold': 5,
                'n_jobs': 1,  # Use single job to avoid multiprocessing issues
                'chunk_size': 30000
            }
        elif sorter_name == 'spykingcircus':
            params = {
                'detect_threshold': 6,
                'template_width_ms': 3,
                'filter': True,
                'merge_spikes': True
            }
        elif sorter_name == 'mountainsort5':
            params = {
                'detect_threshold': 5.5,
                'npca_per_channel': 3,
                'npca_per_subdivision': 10
            }
        elif sorter_name == 'kilosort2_5':
            params = {
                'detect_threshold': 6,
                'projection_threshold': [10, 4],
                'preclust_threshold': 8,
                'car': True,
                'minFR': 0.1,
                'minfr_goodchannels': 0.1
            }

        return params

    def calculate_unit_metrics(self, recording, sorting, output_dir):
        """Calculate comprehensive unit metrics using SpikeInterface"""
        try:
            self.log_message.emit("Calculating unit metrics...")

            unit_ids = sorting.get_unit_ids()
            if len(unit_ids) == 0:
                self.log_message.emit("No units found in sorting results")
                return None, None

            # Create sorting analyzer
            analyzer = si.create_sorting_analyzer(
                sorting=sorting,
                recording=recording,
                folder=output_dir / "sorting_analyzer",
                overwrite=True,
                sparse=True,
                format="binary_folder"
            )

            # Compute extensions in order
            analyzer.compute("random_spikes", seed=42)
            analyzer.compute("waveforms")
            analyzer.compute("templates")
            analyzer.compute("noise_levels")

            # Compute template metrics (includes trough-to-peak calculations)
            analyzer.compute("template_metrics", 
                            include_multi_channel_metrics=True,
                            metric_names=['peak_to_valley', 'halfwidth', 'peak_trough_ratio', 
                                        'recovery_slope', 'repolarization_slope', 'trough_to_peak'])

            # Compute quality metrics
            analyzer.compute("quality_metrics",
                            metric_names=['snr', 'isi_violation', 'firing_rate',
                                        'presence_ratio', 'amplitude_cutoff'])

            # Get computed metrics
            template_metrics = analyzer.get_extension("template_metrics").get_data()
            quality_metrics = analyzer.get_extension("quality_metrics").get_data()

            # Combine all metrics
            all_metrics = pd.concat([quality_metrics, template_metrics], axis=1)

            return all_metrics, analyzer

        except Exception as e:
            self.log_message.emit(f"Error calculating metrics: {str(e)}")
            return None, None

    def generate_unit_reports(self, recording, sorting, analyzer, metrics, output_dir):
        """Generate comprehensive reports using SpikeInterface export functionality"""
        try:
            self.log_message.emit("Generating unit reports...")

            # Compute additional extensions needed for reports
            analyzer.compute(['spike_amplitudes', 'correlograms', 'template_similarity'])

            # Export comprehensive report
            from spikeinterface.exporters import export_report

            reports_dir = output_dir / "spikeinterface_report"

            export_report(
                sorting_analyzer=analyzer,
                output_folder=str(reports_dir),
                format="png",
                peak_sign="neg"
            )

            self.log_message.emit(f"Comprehensive unit reports saved to {reports_dir}")

            # Also generate custom summary plots for each unit
            self.generate_custom_unit_plots(recording, sorting, analyzer, metrics, output_dir)

        except Exception as e:
            self.log_message.emit(f"Error generating unit reports: {str(e)}")



    def run(self):
        try:
            total_recordings = len(self.recordings)

            for idx, (recording, raw_recording, tdt_data, file_path) in enumerate(
                zip(self.recordings, self.raw_recordings, self.tdt_data_objects, self.file_paths)):

                self.log_message.emit(f"\n--- Sorting recording {idx+1}/{total_recordings}: {file_path} ---")

                # Create spike_sort directory in TDT folder
                tdt_path = Path(file_path)
                spike_sort_dir = tdt_path / "spike_sort"
                spike_sort_dir.mkdir(exist_ok=True)

                self.log_message.emit(f"Output directory: {spike_sort_dir}")

                # Get sorter parameters
                self.log_message.emit(f"Using parameters: {self.sorter_params}")

                # Run spike sorting
                self.log_message.emit(f"Running {self.sorter_name} spike sorter...")

                try:
                    # Check if sorter is available
                    available_sorters = si.installed_sorters()
                    if self.sorter_name not in available_sorters:
                        self.log_message.emit(f"Sorter {self.sorter_name} not available. Available sorters: {available_sorters}")
                        continue

                    sorting = si.run_sorter(
                        self.sorter_name,
                        recording,
                        output_folder=spike_sort_dir / "sorter_output",
                        remove_existing_folder=True,
                        verbose=True,
                        **self.sorter_params
                    )

                    unit_ids = sorting.get_unit_ids()
                    self.log_message.emit(f"Sorting completed. Found {len(unit_ids)} units: {unit_ids}")

                    if len(unit_ids) == 0:
                        self.log_message.emit("No units detected by sorter")
                        continue

                    # Calculate metrics
                    metrics, waveforms = self.calculate_unit_metrics(recording, sorting, spike_sort_dir)

                    if metrics is not None and not metrics.empty:
                        # Save metrics
                        metrics_path = spike_sort_dir / "unit_metrics.csv"
                        metrics.to_csv(metrics_path)
                        self.log_message.emit(f"Metrics saved to {metrics_path}")

                        # Generate unit reports
                        if waveforms is not None:
                            self.generate_unit_reports(recording, sorting, waveforms, metrics, spike_sort_dir)

                        # Save sorting results
                        sorting_path = spike_sort_dir / "sorting_results"
                        sorting.save(folder=sorting_path,overwrite=True)
                        self.log_message.emit(f"Sorting results saved to {sorting_path}")

                        # Save recording info
                        recording_info = {
                            'sampling_rate': float(recording.get_sampling_frequency()),
                            'num_channels': int(recording.get_num_channels()),
                            'num_frames': int(recording.get_num_frames()),
                            'duration_sec': float(recording.get_num_frames() / recording.get_sampling_frequency()),
                            'probe_type': getattr(self, 'probe_type', 'Unknown'),
                            'sorter_used': self.sorter_name,
                            'sorter_params': self.sorter_params,
                            'num_units_found': len(unit_ids)
                        }

                        info_path = spike_sort_dir / "recording_info.json"
                        with open(info_path, 'w') as f:
                            json.dump(recording_info, f, indent=2)

                        self.sorting_results.append({
                            'recording_idx': idx,
                            'file_path': str(file_path),
                            'output_dir': str(spike_sort_dir),
                            'num_units': len(unit_ids),
                            'sorting': sorting,
                            'metrics': metrics
                        })

                    else:
                        self.log_message.emit("Failed to calculate metrics or no units found")

                except Exception as e:
                    self.log_message.emit(f"Sorting failed for {file_path}: {str(e)}")
                    # Log more detailed error information
                    import traceback
                    self.log_message.emit(f"Detailed error: {traceback.format_exc()}")
                    continue

                # Update progress
                progress = int((idx + 1) / total_recordings * 100)
                self.progress.emit(progress)

            if self.sorting_results:
                self.finished.emit(True, f"Sorting completed for {len(self.sorting_results)} recordings")
            else:
                self.finished.emit(False, "All sorting attempts failed")

        except Exception as e:
            self.finished.emit(False, f"Sorting error: {str(e)}")


class StreamSelectionDialog(QDialog):
    def __init__(self, available_streams, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select TDT Stream")
        self.setModal(True)
        self.setFixedSize(300, 150)

        self.selected_stream = None

        layout = QVBoxLayout()

        # Stream selection
        stream_layout = QHBoxLayout()
        stream_layout.addWidget(QLabel("Stream:"))

        self.stream_combo = QComboBox()
        self.stream_combo.addItems(available_streams)
        stream_layout.addWidget(self.stream_combo)

        layout.addLayout(stream_layout)

        # Buttons
        button_layout = QHBoxLayout()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_selected_stream(self):
        return self.stream_combo.currentText()


class EPOCSelectionDialog(QDialog):
    def __init__(self, available_epocs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select EPOCs to Save")
        self.setModal(True)
        self.resize(400, 300)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select EPOCs to extract:"))

        self.epoc_list = QListWidget()
        self.epoc_list.setSelectionMode(QAbstractItemView.MultiSelection)

        for epoc in available_epocs:
            self.epoc_list.addItem(epoc)

        layout.addWidget(self.epoc_list)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected_epocs(self):
        return [item.text() for item in self.epoc_list.selectedItems()]

class SorterSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Spike Sorter")
        self.setModal(True)
        self.resize(400, 200)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select spike sorter:"))

        self.sorter_combo = QComboBox()

        # Get available sorters from spikeinterface
        try:
            available_sorters = si.installed_sorters()
            self.sorter_combo.addItems(available_sorters)
        except:
            # Fallback list if installed_sorters fails
            common_sorters = ['spykingcircus', 'mountainsort5', 'tridesclous2', 'kilosort2_5']
            self.sorter_combo.addItems(common_sorters)

        layout.addWidget(self.sorter_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected_sorter(self):
        return self.sorter_combo.currentText()


class AnalysisWorker(QThread):
    finished = Signal(bool, str)
    log_message = Signal(str)
    progress = Signal(int)

    def __init__(self, file_list, output_filename="analysis_results"):
        super().__init__()
        self.file_list = file_list
        self.output_filename = output_filename
        self.processed_blocks = []
        self.failed_blocks = {}

    def get_rat_id_from_tank(self, file_name, t1=None):
        """Extract rat ID from filename - adjust this based on your naming convention"""
        import re
        # Example pattern - adjust based on your file naming
        match = re.search(r'(\d+)', file_name)
        if match:
            return match.group(1)
        return 'NA'

    def get_trial_onsets(self, tdt_data, dB_level):
        """Extract trial onsets for specific dB level"""
        try:
            # Adjust these based on your EPOC structure
            if hasattr(tdt_data.epocs, 'Levl'):
                epoc_data = tdt_data.epocs.Levl.data
                epoc_onsets = tdt_data.epocs.Levl.onset

                # Filter for specific dB level
                mask = epoc_data == dB_level
                return epoc_onsets[mask]
            else:
                return np.array([])
        except Exception as e:
            self.log_message.emit(f"Error getting trial onsets for dB {dB_level}: {str(e)}")
            return np.array([])

    def run(self):
        try:
            # Define parameters
            REF_EPOC = 'Levl'
            Lev_EPOC = 'Levl'
            TRANGE = [-0.05, 0.15]
            dB_ls = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]
            freq_ls = [0]

            # Create empty dataframe to store all units data
            all_units_df = pd.DataFrame()
            total_blocks = len(self.file_list)

            for idx, block in enumerate(self.file_list):
                self.log_message.emit(f"Processing block {idx+1}/{total_blocks}: {block}")

                block_path = block
                sort_path = block + '/spike_sort'

                # Check if spike_sort folder exists
                if not os.path.exists(sort_path):
                    self.log_message.emit(f"No spike_sort folder found in {block}")
                    self.failed_blocks[block] = "No spike_sort folder found"
                    continue

                # Determine genotype
                if 'KO' in block_path:
                    genotype = 'KO'
                elif 'WT' in block_path:
                    genotype = 'WT'
                else:
                    genotype = 'Unknown'

                # Extract rat ID
                rat_id = self.get_rat_id_from_tank(block_path)

                try:
                    # Read the TDT block data for epocs
                    tdt_data = tdt.read_block(block_path, evtype=['epocs'])
                    self.processed_blocks.append(block)

                    try:
                        # Try to read from sorter_output first (where the actual sorting results are)
                        sorter_output_path = os.path.join(sort_path, 'sorter_output')
                        sorting_results_path = os.path.join(sort_path, 'sorting_results')

                        if os.path.exists(sorter_output_path):
                            self.log_message.emit(f"Reading sorting results from sorter_output: {sorter_output_path}")
                            current_sort = si.read_sorter_folder(sorter_output_path)
                        elif os.path.exists(sorting_results_path):
                            self.log_message.emit(f"Reading sorting results from sorting_results: {sorting_results_path}")
                            current_sort = si.read_sorter_folder(sorting_results_path)
                        else:
                            self.log_message.emit(f"No sorting results found in {sort_path}")
                            self.failed_blocks[block] = "No sorting results found in sorter_output or sorting_results"
                            continue

                    except Exception as e:
                        self.log_message.emit(f"Error reading sorting results: {str(e)}")
                        self.failed_blocks[block] = f"Error reading sorting results: {str(e)}"
                        continue
                    unit_ids = current_sort.get_unit_ids()

                    for unit in unit_ids:
                        self.log_message.emit(f"Processing unit: {unit}")

                        # Get spike times for the current unit (convert to seconds)
                        spike_times = current_sort.get_unit_spike_train(unit) / 24414.0625

                        # Create dataframe for this unit's data across all dB levels
                        unit_df = pd.DataFrame(index=range(len(dB_ls)), columns=range(len(freq_ls)))

                        for i, dB in enumerate(dB_ls):
                            # Get trial onsets for this dB level
                            trial_onsets = self.get_trial_onsets(tdt_data, dB)

                            if len(trial_onsets) == 0:
                                unit_df.at[i, 0] = np.array([], dtype=object)
                                continue

                            all_ts = []  # List to store spike times for all trials

                            for onset in trial_onsets:
                                # Filter spikes within the time range relative to trial onset
                                mask = (spike_times >= onset + TRANGE[0]) & (spike_times <= onset + TRANGE[1])
                                trial_spikes = spike_times[mask]

                                # Normalize spike times relative to trial onset
                                normalized_spikes = trial_spikes - onset
                                all_ts.append(normalized_spikes)

                            # Store the spike times for this dB level
                            unit_df.at[i, 0] = np.array(all_ts, dtype=object)

                        # Add metadata columns
                        unit_df['file'] = block_path
                        unit_df['unit_id'] = unit
                        unit_df['genotype'] = genotype
                        unit_df['rat_id'] = rat_id
                        unit_df['dB'] = dB_ls

                        # Calculate pre/post stimulus statistics
                        for i, dB in enumerate(dB_ls):
                            spike_arrays = unit_df.at[i, 0]

                            if len(spike_arrays) == 0:
                                unit_df.at[i, 'pre_fr'] = np.nan
                                unit_df.at[i, 'post_fr'] = np.nan
                                unit_df.at[i, 'delta_fr'] = np.nan
                                unit_df.at[i, 'p_value'] = np.nan
                                unit_df.at[i, 'significant'] = 0
                                continue

                            neg_counts = []  # pre-stimulus
                            pos_counts = []  # post-stimulus

                            for trial_spikes in spike_arrays:
                                # Count spikes in pre-stimulus window
                                pre_count = np.sum((trial_spikes >= -0.05) & (trial_spikes < 0))
                                neg_counts.append(pre_count)

                                # Count spikes in post-stimulus window
                                post_count = np.sum((trial_spikes >= 0) & (trial_spikes <= 0.150))
                                pos_counts.append(post_count)

                            # Convert to firing rates (spikes/sec)
                            pre_fr = [count / 0.05 for count in neg_counts]
                            post_fr = [count / 0.150 for count in pos_counts]

                            # Perform t-test
                            if len(pre_fr) > 1 and len(post_fr) > 1:
                                ttest_result = ttest_ind(pre_fr, post_fr)
                                p_value = ttest_result[1]
                            else:
                                p_value = np.nan

                            # Calculate average change in firing rate
                            avg_dfr = np.mean(np.array(post_fr) - np.array(pre_fr))

                            # Store results
                            unit_df.at[i, 'pre_fr'] = np.mean(pre_fr)
                            unit_df.at[i, 'post_fr'] = np.mean(post_fr)
                            unit_df.at[i, 'delta_fr'] = avg_dfr
                            unit_df.at[i, 'p_value'] = p_value
                            unit_df.at[i, 'significant'] = 1 if p_value < 0.01 else 0

                        # Append to the main dataframe
                        all_units_df = pd.concat([all_units_df, unit_df], ignore_index=True)

                except Exception as e:
                    self.log_message.emit(f"Error processing {sort_path}: {e}")
                    self.failed_blocks[block] = str(e)
                    import traceback
                    traceback.print_exc()

                # Update progress
                progress = int((idx + 1) / total_blocks * 100)
                self.progress.emit(progress)

            # Rename columns for clarity
            all_units_df.rename(columns={0: 'spike_trains'}, inplace=True)

            # Save results
            csv_path = f'{self.output_filename}.csv'
            pkl_path = f'{self.output_filename}.pkl'

            all_units_df.to_csv(csv_path, index=False)

            with open(pkl_path, 'wb') as f:
                pickle.dump(all_units_df, f)

            # Save processing log
            with open('processing_log.txt', 'w') as f:
                f.write(f"Total blocks: {len(self.file_list)}\n")
                f.write(f"Processed blocks: {len(self.processed_blocks)}\n")
                f.write(f"Failed blocks: {len(self.failed_blocks)}\n")
                f.write("\nFailed blocks with errors:\n")
                for block, error in self.failed_blocks.items():
                    f.write(f"{block}: {error}\n")

            self.finished.emit(True, f"Analysis completed. Processed {len(self.processed_blocks)} blocks. Results saved as {csv_path} and {pkl_path}")

        except Exception as e:
            self.finished.emit(False, f"Analysis error: {str(e)}")


class SorterParameterDialog(QDialog):
    def __init__(self, sorter_name, parent=None):
        super().__init__(parent)
        self.sorter_name = sorter_name
        self.setWindowTitle(f"Edit Parameters for {sorter_name}")
        self.setModal(True)
        self.resize(500, 600)

        # Get default parameters
        try:
            self.default_params = si.get_default_sorter_params(sorter_name)
        except Exception as e:
            self.default_params = {}

        self.param_widgets = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel(f"Parameters for {self.sorter_name}")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(title)

        # Scrollable area for parameters
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        if not self.default_params:
            scroll_layout.addWidget(QLabel("No parameters available for this sorter"))
        else:
            # Create widgets for each parameter
            for param_name, param_value in self.default_params.items():
                param_layout = self.create_parameter_widget(param_name, param_value)
                scroll_layout.addLayout(param_layout)

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        # Reset to defaults button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        layout.addWidget(reset_btn)

        

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def create_parameter_widget(self, param_name, param_value):
        """Create appropriate widget based on parameter type"""
        layout = QHBoxLayout()

        # Parameter name label
        name_label = QLabel(f"{param_name}:")
        name_label.setMinimumWidth(150)
        name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(name_label)

        # Create appropriate input widget based on value type
        if isinstance(param_value, bool):
            widget = QCheckBox()
            widget.setChecked(param_value)
            self.param_widgets[param_name] = widget

        elif isinstance(param_value, int):
            widget = QSpinBox()
            widget.setRange(-999999, 999999)
            widget.setValue(param_value)
            self.param_widgets[param_name] = widget

        elif isinstance(param_value, float):
            widget = QDoubleSpinBox()
            widget.setRange(-999999.0, 999999.0)
            widget.setDecimals(6)
            widget.setValue(param_value)
            self.param_widgets[param_name] = widget

        elif isinstance(param_value, str):
            widget = QLineEdit()
            widget.setText(param_value)
            self.param_widgets[param_name] = widget

        elif isinstance(param_value, (list, tuple)):
            widget = QLineEdit()
            widget.setText(str(param_value))
            widget.setToolTip("Enter as Python list/tuple format, e.g., [1, 2, 3]")
            self.param_widgets[param_name] = widget

        elif param_value is None:
            widget = QLineEdit()
            widget.setText("None")
            widget.setToolTip("Enter 'None' for null value or appropriate value")
            self.param_widgets[param_name] = widget

        else:
            # For any other type, use text input
            widget = QLineEdit()
            widget.setText(str(param_value))
            self.param_widgets[param_name] = widget

        layout.addWidget(widget)

        # Default value label
        default_label = QLabel(f"(default: {param_value})")
        default_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(default_label)

        return layout

    def reset_to_defaults(self):
        """Reset all parameters to their default values"""
        for param_name, param_value in self.default_params.items():
            if param_name in self.param_widgets:
                widget = self.param_widgets[param_name]

                if isinstance(widget, QCheckBox):
                    widget.setChecked(param_value)
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(param_value)
                elif isinstance(widget, QLineEdit):
                    if param_value is None:
                        widget.setText("None")
                    else:
                        widget.setText(str(param_value))

    def get_parameters(self):
        """Get the current parameter values from the widgets"""
        params = {}

        for param_name, widget in self.param_widgets.items():
            try:
                if isinstance(widget, QCheckBox):
                    params[param_name] = widget.isChecked()

                elif isinstance(widget, QSpinBox):
                    params[param_name] = widget.value()

                elif isinstance(widget, QDoubleSpinBox):
                    params[param_name] = widget.value()

                elif isinstance(widget, QLineEdit):
                    text = widget.text().strip()

                    # Handle None values
                    if text.lower() == 'none':
                        params[param_name] = None
                    else:
                        # Try to evaluate as Python literal (for lists, tuples, etc.)
                        try:
                            # First try to parse as literal
                            import ast
                            params[param_name] = ast.literal_eval(text)
                        except (ValueError, SyntaxError):
                            # If that fails, treat as string
                            params[param_name] = text

            except Exception as e:
                # If conversion fails, use the original default value
                params[param_name] = self.default_params.get(param_name)

        return params

class ProbeGeometryWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 300)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel("Probe Geometry")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        # Canvas for probe plot
        self.canvas = FigureCanvas(Figure(figsize=(3, 3), dpi=80))
        self.canvas.figure.patch.set_facecolor('white')
        layout.addWidget(self.canvas)

        self.probe = None
        self.clear_plot()

    def clear_plot(self):
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'No probe\nconfigured', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.canvas.draw()

    def update_probe_plot(self, probe):
        self.probe = probe
        if probe is None:
            self.clear_plot()
            return

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        # Get probe positions and plot
        positions = probe.contact_positions

        # Plot contact positions
        ax.scatter(positions[:, 0], positions[:, 1], 
                  c='red', s=30, alpha=0.8, edgecolors='black', linewidth=0.5)

        # Add channel numbers
        for i, pos in enumerate(positions):
            ax.annotate(str(i+1), (pos[0], pos[1]), 
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=6, ha='left', va='bottom')

        # Set equal aspect ratio and clean up axes
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (μm)', fontsize=8)
        ax.set_ylabel('Y (μm)', fontsize=8)
        ax.tick_params(labelsize=6)

        # Add some padding around the probe
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        padding = max(x_range, y_range) * 0.1 + 10

        ax.set_xlim(positions[:, 0].min() - padding, positions[:, 0].max() + padding)
        ax.set_ylim(positions[:, 1].min() - padding, positions[:, 1].max() + padding)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

class ProbeConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Probe Configuration")
        self.setModal(True)
        self.resize(400, 300)

        layout = QVBoxLayout(self)

        # Probe type selection
        layout.addWidget(QLabel("Select Probe Type:"))
        self.probe_combo = QComboBox()
        self.probe_combo.addItems(["Linear 16-channel", "Tetrode 32-channel", "Custom"])
        layout.addWidget(self.probe_combo)

        # Description area
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setMaximumHeight(150)
        layout.addWidget(self.description_text)

        # Update description when selection changes
        self.probe_combo.currentTextChanged.connect(self.update_description)
        self.update_description()

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def update_description(self):
        probe_type = self.probe_combo.currentText()
        descriptions = {
            "Linear 16-channel": """
Linear probe with 16 channels arranged vertically:
- 16 contacts in a single column
- 100μm spacing between contacts
- Channel mapping: [9,8,10,7,13,4,12,5,15,2,16,1,14,3,11,6]
- Suitable for cortical recordings
            """,
            "Tetrode 32-channel": """
Tetrode probe with 32 channels in 8 bundles:
- 8 tetrode bundles (4 channels each)
- 2x4 bundle arrangement
- 200μm spacing between bundles
- 25μm tetrode tip separation
- Suitable for hippocampal recordings
            """,
            "Custom": """
Custom probe configuration:
- Load probe from .prb file
- Or define custom geometry
- Specify your own channel mapping
            """
        }
        self.description_text.setText(descriptions.get(probe_type, ""))

    def get_selected_probe(self):
        return self.probe_combo.currentText()
class UnitQualityDialog(QDialog):
    def __init__(self, unit_reports_dir, parent=None):
        super().__init__(parent)
        self.unit_reports_dir = Path(unit_reports_dir)
        self.unit_png_files = []
        self.current_unit_idx = 0
        self.quality_data = []

        # Find all unit PNG files
        self.find_unit_pngs()

        if not self.unit_png_files:
            self.close()
            return

        self.setWindowTitle(f"Unit Quality Assessment - {len(self.unit_png_files)} units")
        self.setModal(True)
        self.resize(1000, 700)

        self.init_ui()
        self.load_current_unit()

    def find_unit_pngs(self):
        """Find all unit PNG files in the spikeinterface_report/units folder"""
        units_dir = self.unit_reports_dir / "units"

        if not units_dir.exists():
            print(f"Units directory not found: {units_dir}")
            return

        # Look for numbered PNG files (0.png, 1.png, 2.png, etc.)
        png_files = []
        unit_id = 0

        while True:
            png_file = units_dir / f"{unit_id}.png"
            if png_file.exists():
                png_files.append(png_file)
                unit_id += 1
            else:
                break

        self.unit_png_files = png_files
        print(f"Found {len(self.unit_png_files)} unit PNG files")

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Progress info
        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-weight: bold; font-size: 16px; margin: 10px;")
        layout.addWidget(self.progress_label)

        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray; background-color: white;")
        self.image_label.setMinimumSize(800, 500)

        # Scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        # Quality buttons
        button_layout = QHBoxLayout()

        self.noise_btn = QPushButton("NOISE")
        self.noise_btn.setStyleSheet("""
            QPushButton { 
                background-color: #ff4757; 
                color: white; 
                font-weight: bold; 
                font-size: 14px;
                padding: 15px 30px;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #ff3742; }
        """)
        self.noise_btn.clicked.connect(lambda: self.rate_unit("noise"))

        self.mua_btn = QPushButton("MUA")
        self.mua_btn.setStyleSheet("""
            QPushButton { 
                background-color: #ffa502; 
                color: white; 
                font-weight: bold; 
                font-size: 14px;
                padding: 15px 30px;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #ff9500; }
        """)
        self.mua_btn.clicked.connect(lambda: self.rate_unit("mua"))

        self.good_btn = QPushButton("GOOD")
        self.good_btn.setStyleSheet("""
            QPushButton { 
                background-color: #2ed573; 
                color: white; 
                font-weight: bold; 
                font-size: 14px;
                padding: 15px 30px;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #26d0ce; }
        """)
        self.good_btn.clicked.connect(lambda: self.rate_unit("good"))

        button_layout.addStretch()
        button_layout.addWidget(self.noise_btn)
        button_layout.addWidget(self.mua_btn)
        button_layout.addWidget(self.good_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        # Navigation buttons
        nav_layout = QHBoxLayout()

        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.setStyleSheet("padding: 8px 16px; font-size: 12px;")
        self.prev_btn.clicked.connect(self.previous_unit)

        self.skip_btn = QPushButton("Skip")
        self.skip_btn.setStyleSheet("padding: 8px 16px; font-size: 12px;")
        self.skip_btn.clicked.connect(self.skip_unit)

        self.finish_btn = QPushButton("Finish & Save")
        self.finish_btn.setStyleSheet("padding: 8px 16px; font-size: 12px; font-weight: bold;")
        self.finish_btn.clicked.connect(self.finish_assessment)

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.skip_btn)
        nav_layout.addWidget(self.finish_btn)

        layout.addLayout(nav_layout)

    def load_current_unit(self):
        """Load and display the current unit's PNG report"""
        if self.current_unit_idx >= len(self.unit_png_files):
            self.finish_assessment()
            return

        current_file = self.unit_png_files[self.current_unit_idx]
        unit_id = current_file.stem  # This will be "0", "1", "2", etc.

        # Update progress
        self.progress_label.setText(f"Unit {unit_id} ({self.current_unit_idx + 1} of {len(self.unit_png_files)})")

        # Load and display image
        try:
            from qtpy.QtGui import QPixmap
            pixmap = QPixmap(str(current_file))
            if not pixmap.isNull():
                # Scale image to fit while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText(f"Could not load image:\n{current_file}")
        except Exception as e:
            self.image_label.setText(f"Error loading image:\n{str(e)}")

        # Update button states
        self.prev_btn.setEnabled(self.current_unit_idx > 0)

    def rate_unit(self, quality):
        """Rate the current unit and move to next"""
        current_file = self.unit_png_files[self.current_unit_idx]
        unit_id = current_file.stem  # This will be "0", "1", "2", etc.

        # Store quality rating
        self.quality_data.append({
            'unit_id': int(unit_id),
            'quality': quality,
            'file_path': str(current_file)
        })

        print(f"Rated unit {unit_id} as {quality}")

        # Move to next unit
        self.next_unit()

    def skip_unit(self):
        """Skip current unit without rating"""
        current_file = self.unit_png_files[self.current_unit_idx]
        unit_id = current_file.stem
        print(f"Skipped unit {unit_id}")
        self.next_unit()

    def next_unit(self):
        """Move to next unit"""
        self.current_unit_idx += 1
        self.load_current_unit()

    def previous_unit(self):
        """Go back to previous unit"""
        if self.current_unit_idx > 0:
            self.current_unit_idx -= 1

            # Remove the previous rating if it exists
            current_file = self.unit_png_files[self.current_unit_idx]
            unit_id = int(current_file.stem)

            # Remove any existing rating for this unit
            self.quality_data = [item for item in self.quality_data if item['unit_id'] != unit_id]

            self.load_current_unit()

    def finish_assessment(self):
        """Save results and close dialog"""
        if self.quality_data:
            self.save_quality_data()
            print(f"Assessment completed. Rated {len(self.quality_data)} units.")
        else:
            print("No units were rated.")
        self.accept()

    def save_quality_data(self):
        """Save quality assessments to CSV"""
        try:
            df = pd.DataFrame(self.quality_data)
            # Sort by unit_id to maintain order
            df = df.sort_values('unit_id')

            output_path = self.unit_reports_dir.parent / "unit_quality.csv"
            df.to_csv(output_path, index=False)
            print(f"Unit quality data saved to: {output_path}")
        except Exception as e:
            print(f"Error saving quality data: {str(e)}")

    def get_quality_data(self):
        """Return the collected quality data"""
        return self.quality_data


    def load_current_unit(self):
        """Load and display the current unit's PNG report"""
        if self.current_unit_idx >= len(self.unit_png_files):
            self.finish_assessment()
            return

        current_file = self.unit_png_files[self.current_unit_idx]

        # Update progress
        self.progress_label.setText(f"Unit {self.current_unit_idx + 1} of {len(self.unit_png_files)}")

        # Load and display image
        try:
            from qtpy.QtGui import QPixmap
            pixmap = QPixmap(str(current_file))
            if not pixmap.isNull():
                self.image_label.setPixmap(pixmap)
            else:
                self.image_label.setText(f"Could not load image:\n{current_file}")
        except Exception as e:
            self.image_label.setText(f"Error loading image:\n{str(e)}")

        # Update button states
        self.prev_btn.setEnabled(self.current_unit_idx > 0)

    def rate_unit(self, quality):
        """Rate the current unit and move to next"""
        current_file = self.unit_png_files[self.current_unit_idx]

        # Extract unit ID from filename
        unit_id = self.extract_unit_id(current_file)

        # Store quality rating
        self.quality_data.append({
            'unit_id': unit_id,
            'quality': quality,
            'file_path': str(current_file)
        })

        # Move to next unit
        self.next_unit()

    def extract_unit_id(self, file_path):
        """Extract unit ID from filename"""
        filename = file_path.name
        # Try to extract unit ID from filename patterns like "unit_001.png" or "unit001.png"
        import re
        match = re.search(r'unit[_]?(\d+)', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        else:
            # Fallback to using the index
            return self.current_unit_idx

    def skip_unit(self):
        """Skip current unit without rating"""
        self.next_unit()

    def next_unit(self):
        """Move to next unit"""
        self.current_unit_idx += 1
        self.load_current_unit()

    def previous_unit(self):
        """Go back to previous unit"""
        if self.current_unit_idx > 0:
            self.current_unit_idx -= 1
            # Remove the previous rating if it exists
            if self.quality_data and self.quality_data[-1]['unit_id'] == self.extract_unit_id(self.unit_png_files[self.current_unit_idx]):
                self.quality_data.pop()
            self.load_current_unit()

    def finish_assessment(self):
        """Save results and close dialog"""
        if self.quality_data:
            self.save_quality_data()
        self.accept()

    def save_quality_data(self):
        """Save quality assessments to CSV"""
        try:
            df = pd.DataFrame(self.quality_data)
            output_path = self.unit_reports_dir.parent / "unit_quality.csv"
            df.to_csv(output_path, index=False)
            print(f"Unit quality data saved to: {output_path}")
        except Exception as e:
            print(f"Error saving quality data: {str(e)}")

    def get_quality_data(self):
        """Return the collected quality data"""
        return self.quality_data

    def load_current_unit(self):
        """Load and display the current unit's PNG report"""
        if self.current_unit_idx >= len(self.unit_png_files):
            self.finish_assessment()
            return

        current_file = self.unit_png_files[self.current_unit_idx]

        # Update progress
        self.progress_label.setText(f"Unit {self.current_unit_idx + 1} of {len(self.unit_png_files)}")

        # Load and display image
        try:
            from qtpy.QtGui import QPixmap
            pixmap = QPixmap(str(current_file))
            if not pixmap.isNull():
                self.image_label.setPixmap(pixmap)
            else:
                self.image_label.setText(f"Could not load image:\n{current_file}")
        except Exception as e:
            self.image_label.setText(f"Error loading image:\n{str(e)}")

        # Update button states
        self.prev_btn.setEnabled(self.current_unit_idx > 0)

    def rate_unit(self, quality):
        """Rate the current unit and move to next"""
        current_file = self.unit_png_files[self.current_unit_idx]

        # Extract unit ID from filename
        unit_id = self.extract_unit_id(current_file)

        # Store quality rating
        self.quality_data.append({
            'unit_id': unit_id,
            'quality': quality,
            'file_path': str(current_file)
        })

        # Move to next unit
        self.next_unit()

    def extract_unit_id(self, file_path):
        """Extract unit ID from filename"""
        filename = file_path.name
        # Try to extract unit ID from filename patterns like "unit_001.png" or "unit001.png"
        import re
        match = re.search(r'unit[_]?(\d+)', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        else:
            # Fallback to using the index
            return self.current_unit_idx

    def skip_unit(self):
        """Skip current unit without rating"""
        self.next_unit()

    def next_unit(self):
        """Move to next unit"""
        self.current_unit_idx += 1
        self.load_current_unit()

    def previous_unit(self):
        """Go back to previous unit"""
        if self.current_unit_idx > 0:
            self.current_unit_idx -= 1
            # Remove the previous rating if it exists
            if self.quality_data and self.quality_data[-1]['unit_id'] == self.extract_unit_id(self.unit_png_files[self.current_unit_idx]):
                self.quality_data.pop()
            self.load_current_unit()

    def finish_assessment(self):
        """Save results and close dialog"""
        if self.quality_data:
            self.save_quality_data()
        self.accept()

    def save_quality_data(self):
        """Save quality assessments to CSV"""
        try:
            df = pd.DataFrame(self.quality_data)
            output_path = self.unit_reports_dir.parent / "unit_quality.csv"
            df.to_csv(output_path, index=False)
            print(f"Unit quality data saved to: {output_path}")
        except Exception as e:
            print(f"Error saving quality data: {str(e)}")

    def get_quality_data(self):
        """Return the collected quality data"""
        return self.quality_data

class AnalysisGUI(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PSTH Analysis")
        self.setModal(True)
        self.resize(600, 400)

        self.file_list = []
        self.current_worker = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("PSTH Analysis - Extract Sorted Data")
        title.setStyleSheet("font-weight: bold; font-size: 16px; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # File selection section
        file_section = QVBoxLayout()
        file_section.addWidget(QLabel("Select data files:"))

        file_buttons = QHBoxLayout()
        self.select_files_btn = QPushButton("Select Folders")
        self.load_list_btn = QPushButton("Load File List")
        self.select_files_btn.clicked.connect(self.select_folders)
        self.load_list_btn.clicked.connect(self.load_file_list)

        file_buttons.addWidget(self.select_files_btn)
        file_buttons.addWidget(self.load_list_btn)
        file_section.addLayout(file_buttons)

        # File list display
        self.file_list_widget = QListWidget()
        self.file_list_widget.setMaximumHeight(150)
        file_section.addWidget(self.file_list_widget)

        layout.addLayout(file_section)

        # Output filename
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output filename:"))
        self.output_filename_edit = QLineEdit("analysis_results")
        output_layout.addWidget(self.output_filename_edit)
        layout.addLayout(output_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Log area
        layout.addWidget(QLabel("Processing Log:"))
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Buttons
        button_layout = QHBoxLayout()

        self.analyze_btn = QPushButton("Start Analysis")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setStyleSheet("font-weight: bold; padding: 10px;")

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)

        button_layout.addWidget(self.analyze_btn)
        button_layout.addWidget(self.close_btn)
        layout.addLayout(button_layout)

    def select_folders(self):
        """Select multiple folders containing spike sorted data"""
        folder = QFileDialog.getExistingDirectory(self, "Select Directory Containing Data Folders")
        if folder:
            # Find all subdirectories that contain spike_sort folders
            base_path = Path(folder)
            self.file_list = []

            for item in base_path.iterdir():
                if item.is_dir():
                    spike_sort_path = item / "spike_sort"
                    if spike_sort_path.exists():
                        self.file_list.append(str(item))

            self.update_file_list_display()
            self.add_log_message(f"Found {len(self.file_list)} folders with spike_sort data")

    def load_file_list(self):
        """Load file list from CSV"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                df = pd.read_csv(file_path)
                self.file_list = df.iloc[:, 0].dropna().tolist()
                self.update_file_list_display()
                self.add_log_message(f"Loaded {len(self.file_list)} files from CSV")
            except Exception as e:
                self.add_log_message(f"Error loading CSV: {str(e)}")

    def update_file_list_display(self):
        """Update the file list widget"""
        self.file_list_widget.clear()
        for file_path in self.file_list:
            self.file_list_widget.addItem(Path(file_path).name)

    def start_analysis(self):
        """Start the analysis process"""
        if not self.file_list:
            self.add_log_message("Please select files first")
            return

        output_filename = self.output_filename_edit.text().strip()
        if not output_filename:
            output_filename = "analysis_results"

        self.add_log_message(f"Starting analysis of {len(self.file_list)} files...")

        # Disable button and show progress
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Start worker
        self.current_worker = AnalysisWorker(self.file_list, output_filename)
        self.current_worker.finished.connect(self.on_analysis_finished)
        self.current_worker.log_message.connect(self.add_log_message)
        self.current_worker.progress.connect(self.progress_bar.setValue)
        self.current_worker.start()

    def on_analysis_finished(self, success, message):
        """Handle analysis completion"""
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        if success:
            self.add_log_message("✓ " + message)
        else:
            self.add_log_message("✗ " + message)

        self.current_worker = None

    def add_log_message(self, message):
        """Add message to log"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())



class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.raw_recording = None
        self.processed_recording = None
        self.time_window = 5.0  # seconds
        self.start_time = 0.0
        self.display_mode = "both"  # "raw", "processed", "both"

    def set_display_mode(self, mode):
        self.display_mode = mode

    def plot_recordings(self, raw_recording, processed_recording, start_time=0.0):
        self.raw_recording = raw_recording
        self.processed_recording = processed_recording
        self.start_time = start_time

        self.fig.clear()

        if raw_recording is None or processed_recording is None:
            return

        # Get sampling rate and calculate sample indices
        fs = raw_recording.get_sampling_frequency()
        start_sample = int(start_time * fs)
        end_sample = int((start_time + self.time_window) * fs)

        # Ensure we don't exceed recording bounds
        max_samples = min(raw_recording.get_num_frames(), processed_recording.get_num_frames())
        end_sample = min(end_sample, max_samples)

        if start_sample >= end_sample:
            return

        # Get data chunks
        raw_data = raw_recording.get_traces(start_frame=start_sample, end_frame=end_sample)
        processed_data = processed_recording.get_traces(start_frame=start_sample, end_frame=end_sample)

        # Convert to microvolts if needed
        if raw_data.dtype != np.float32:
            raw_data = raw_data.astype(np.float32)
        if processed_data.dtype != np.float32:
            processed_data = processed_data.astype(np.float32)

        # Time vector
        time_vec = np.linspace(start_time, start_time + self.time_window, raw_data.shape[0])

        # Plot each channel
        n_channels = raw_data.shape[1]

        ax = self.fig.add_subplot(111)

        # Calculate appropriate channel spacing based on data range
        if self.display_mode == "raw":
            data_std = np.std(raw_data, axis=0)
        elif self.display_mode == "processed":
            data_std = np.std(processed_data, axis=0)
        else:  # both
            raw_std = np.std(raw_data, axis=0)
            processed_std = np.std(processed_data, axis=0)
            data_std = np.maximum(raw_std, processed_std)

        max_std = np.max(data_std)

        # Use adaptive spacing
        if max_std > 0:
            channel_spacing = max_std * 8  # 8x standard deviation spacing
        else:
            channel_spacing = 100  # fallback spacing

        print(f"Display mode: {self.display_mode}")
        print(f"Data ranges - Raw: {np.min(raw_data):.2f} to {np.max(raw_data):.2f}")
        print(f"Data ranges - Processed: {np.min(processed_data):.2f} to {np.max(processed_data):.2f}")
        print(f"Channel spacing: {channel_spacing:.2f}")

        legend_added = False

        for ch in range(n_channels):
            offset = ch * channel_spacing

            # Plot based on display mode
            if self.display_mode == "raw":
                ax.plot(time_vec, raw_data[:, ch] + offset, 'k-', linewidth=0.8, 
                       label='Raw' if not legend_added else "")
                legend_added = True

            elif self.display_mode == "processed":
                ax.plot(time_vec, processed_data[:, ch] + offset, 'g-', linewidth=0.8,
                       label='Processed' if not legend_added else "")
                legend_added = True

            else:  # both
                ax.plot(time_vec, raw_data[:, ch] + offset, 'k-', linewidth=0.5, alpha=0.7,
                       label='Raw' if not legend_added else "")
                ax.plot(time_vec, processed_data[:, ch] + offset, 'g-', linewidth=0.8,
                       label='Processed' if not legend_added else "")
                legend_added = True

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel')

        # Set title based on display mode
        mode_text = {
            "raw": "Raw Neural Recordings",
            "processed": "Processed Neural Recordings", 
            "both": "Raw (black) & Processed (green) Neural Recordings"
        }
        ax.set_title(f'{mode_text[self.display_mode]}: {start_time:.1f} - {start_time + self.time_window:.1f} s')

        # Set y-axis labels to channel numbers
        ax.set_yticks([ch * channel_spacing for ch in range(n_channels)])
        ax.set_yticklabels([f'Ch {ch+1}' for ch in range(n_channels)])

        # Add legend
        if legend_added:
            ax.legend(loc='upper right')

        ax.grid(True, alpha=0.3)
        ax.set_xlim(start_time, start_time + self.time_window)

        self.fig.tight_layout()
        self.draw()

class DataLoader(QThread):
    finished = Signal(bool, str)
    log_message = Signal(str)

    def __init__(self, file_paths, probe_type="Linear 16-channel", is_batch=False):
        super().__init__()
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.probe_type = probe_type
        self.is_batch = is_batch
        self.recordings = []
        self.raw_recordings = []
        self.tdt_data_objects = []  # Store TDT data objects for EPOC access

    def setup_linear_probe(self):
        from probeinterface import Probe
        n = 16
        positions = np.zeros((n, 2))
        for i in range(n):
            x = 0
            y = i * 100
            positions[i] = x, y

        probe = Probe(ndim=2, si_units='um')
        probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 5})
        probe_map = np.array([9,8,10,7,13,4,12,5,15,2,16,1,14,3,11,6]) - 1
        probe.set_device_channel_indices(probe_map)
        return probe

    def setup_tetrode_probe(self):
        from probeinterface import Probe

        probe = Probe(ndim=2, si_units='um')

        # Tetrode bundle positions (2 rows of 4, 200um separation)
        bundle_positions = []
        for row in range(2):
            for col in range(4):
                x = col * 200  # 200um between bundles
                y = row * 200  # 200um between rows
                bundle_positions.append((x, y))

        # Generate electrode positions (4 electrodes per tetrode in square formation)
        positions = []
        contact_ids = []
        shank_ids = []

        tetrode_offset = 12.5  # 25um tetrode tip separation / 2

        for bundle_idx, (bx, by) in enumerate(bundle_positions):
            # 4 electrodes per tetrode in square pattern
            tetrode_positions = [
                (bx - tetrode_offset, by - tetrode_offset),  # bottom-left
                (bx + tetrode_offset, by - tetrode_offset),  # bottom-right
                (bx - tetrode_offset, by + tetrode_offset),  # top-left
                (bx + tetrode_offset, by + tetrode_offset)   # top-right
            ]

            for i, (x, y) in enumerate(tetrode_positions):
                positions.append([x, y])
                contact_ids.append(bundle_idx * 4 + i)
                shank_ids.append(bundle_idx)

        probe.set_contacts(positions=np.array(positions), 
                          contact_ids=np.array(contact_ids),
                          shank_ids=np.array(shank_ids),
                          shapes='circle',
                          shape_params={'radius':5})

        # Tetrode channel mapping
        bundle_channels = {
            'A': [5, 3, 1, 7],    # Bundle 0
            'C': [10, 12, 14, 16], # Bundle 1  
            'E': [17, 19, 21, 23], # Bundle 2
            'G': [32, 30, 28, 26], # Bundle 3
            'B': [2, 4, 6, 8],     # Bundle 4
            'D': [15, 13, 11, 9],  # Bundle 5
            'F': [24, 22, 20, 18], # Bundle 6
            'H': [31, 29, 27, 25]  # Bundle 7
        }

        tetrode_tdt_map = np.zeros(32, dtype=int)
        bundle_order = ['A', 'C', 'E', 'G', 'B', 'D', 'F', 'H']

        for bundle_idx, bundle_name in enumerate(bundle_order):
            channels = bundle_channels[bundle_name]
            for ch_idx, channel in enumerate(channels):
                probe_contact_idx = bundle_idx * 4 + ch_idx
                tetrode_tdt_map[probe_contact_idx] = channel - 1  # Convert to 0-based indexing

        probe.set_device_channel_indices(tetrode_tdt_map)
        return probe

    def setup_probe(self):
        if self.probe_type == "Linear 16-channel":
            return self.setup_linear_probe()
        elif self.probe_type == "Tetrode 32-channel":
            return self.setup_tetrode_probe()
        else:  # Custom
            # For now, default to linear probe
            # TODO: Add custom probe loading functionality
            self.log_message.emit("Custom probe not implemented yet, using linear probe")
            return self.setup_linear_probe()

    def process_single_recording(self, file_path):
        try:
            base_folder = Path(file_path)
            tbk_files = list(base_folder.glob("*.Tbk"))

            if not tbk_files:
                return None, None, None, f"No .Tbk file found in {base_folder}"

            tdt_tank = str(tbk_files[0])
            probe = self.setup_probe()

            self.log_message.emit(f"Loading TDT tank: {tdt_tank}")
            self.log_message.emit(f"Using probe type: {self.probe_type}")

            # Load TDT data object for EPOC access (using the tank directory, not the .Tbk file)
            tank_dir = str(base_folder)
            tdt_data = tdt.read_block(tank_dir)

            available_streams = []
            for stream_name in tdt_data.streams.keys():
                if hasattr(tdt_data.streams[stream_name], 'data'):
                    available_streams.append(stream_name)

            if not available_streams:
                return None, None, None, "No valid streams found in TDT file"

            dialog = StreamSelectionDialog(available_streams)
            if dialog.exec_() == QDialog.Accepted:
                selected_stream = dialog.get_selected_stream()
            else:
                return None, None, None, "Stream selection cancelled"

            self.log_message.emit(f"Selected stream: {selected_stream}")

            # Load data with selected stream
            raw_data = si.read_tdt(tdt_tank, stream_name=selected_stream)
            self.log_message.emit(f"Raw data loaded: {raw_data.get_num_channels()} channels, {raw_data.get_num_frames()} samples")

            # Get a small sample to check data range
            sample_data = raw_data.get_traces(start_frame=0, end_frame=min(1000, raw_data.get_num_frames()))
            self.log_message.emit(f"Raw data range: {np.min(sample_data):.2f} to {np.max(sample_data):.2f}")

            recording_filtered = si.bandpass_filter(raw_data, freq_min=300, freq_max=6000, dtype=np.float32)
            self.log_message.emit("Applied bandpass filter (300-6000 Hz)")

            # Check filtered data range
            sample_filtered = recording_filtered.get_traces(start_frame=0, end_frame=min(1000, recording_filtered.get_num_frames()))
            self.log_message.emit(f"Filtered data range: {np.min(sample_filtered):.2f} to {np.max(sample_filtered):.2f}")

            rec_no_bads = recording_filtered.remove_channels([])
            recording_cmr = si.common_reference(rec_no_bads, operator="median", reference="global")
            self.log_message.emit("Applied common median reference")

            # Check CMR data range
            sample_cmr = recording_cmr.get_traces(start_frame=0, end_frame=min(1000, recording_cmr.get_num_frames()))
            self.log_message.emit(f"CMR data range: {np.min(sample_cmr):.2f} to {np.max(sample_cmr):.2f}")

            # Set probe on FINAL recording object and store result
            recording_final = recording_cmr.set_probe(probe)
            self.log_message.emit("Probe geometry applied")

            # Verify probe is attached
            if recording_final.get_probe() is None:
                self.log_message.emit("WARNING: Probe not properly attached!")
            else:
                self.log_message.emit(f"Probe verified: {len(recording_final.get_probe().contact_positions)} contacts")

            duration = recording_final.get_num_frames() / recording_final.get_sampling_frequency()
            self.log_message.emit(f"Recording duration: {duration:.2f} seconds")
            self.log_message.emit(f"Sampling rate: {recording_final.get_sampling_frequency()} Hz")

            # Return the probe-attached recordings
            return raw_data.set_probe(probe), recording_final, tdt_data, "Success"

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.log_message.emit(error_msg)
            return None, None, None, str(e)


    def run(self):
        try:
            self.recordings = []
            self.raw_recordings = []
            self.tdt_data_objects = []
            failed_files = []

            for file_path in self.file_paths:
                self.log_message.emit(f"\n--- Processing: {file_path} ---")
                raw_rec, processed_rec, tdt_data, message = self.process_single_recording(file_path)
                if processed_rec is not None:
                    self.raw_recordings.append(raw_rec)
                    self.recordings.append(processed_rec)
                    self.tdt_data_objects.append(tdt_data)
                    self.log_message.emit("✓ Recording loaded successfully")
                else:
                    failed_files.append(f"{file_path}: {message}")
                    self.log_message.emit("✗ Recording failed to load")

            if self.recordings:
                if failed_files:
                    message = f"Loaded {len(self.recordings)} files. Failed: {len(failed_files)}"
                else:
                    message = f"Successfully loaded {len(self.recordings)} recording(s)"
                self.finished.emit(True, message)
            else:
                self.finished.emit(False, f"All files failed to load: {'; '.join(failed_files)}")

        except Exception as e:
            self.finished.emit(False, f"Loading error: {str(e)}")

class SortingPipelineGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recordings = []
        self.raw_recordings = []
        self.tdt_data_objects = []
        self.file_paths = []
        self.current_loader = None
        self.current_sorting_worker = None
        self.current_recording_idx = 0
        self.probe_type = "Linear 16-channel"
        self.current_probe = None
        self.selected_sorter = "simple"
        self.sorting_params = {}
        self.sorting_results = []
        # Add these after existing variables
        self.batch_file_list = []
        self.current_batch_index = 0
        self.is_batch_mode = False

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Auerbach Lab Sorting Pipeline")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Top section with buttons and plot
        top_layout = QHBoxLayout()

        # Left panel with buttons and probe geometry
        left_layout = QVBoxLayout()

        self.manual_select_btn = QPushButton("Manually Select Files")
        self.load_list_btn = QPushButton("Load File List")
        self.probe_config_btn = QPushButton("Configure Probe")

        self.manual_select_btn.clicked.connect(self.manual_select_files)
        self.load_list_btn.clicked.connect(self.load_file_list)
        self.probe_config_btn.clicked.connect(self.configure_probe)

        left_layout.addWidget(self.manual_select_btn)
        left_layout.addWidget(self.load_list_btn)
        left_layout.addWidget(self.probe_config_btn)

        # Add probe geometry widget
        self.probe_geometry_widget = ProbeGeometryWidget()
        left_layout.addWidget(self.probe_geometry_widget)

        left_layout.addStretch()

        # Center plot area
        plot_layout = QVBoxLayout()

        self.plot_canvas = PlotCanvas(self, width=8, height=6)
        plot_layout.addWidget(self.plot_canvas)

        # Controls layout (slider and toggle button)
        controls_layout = QVBoxLayout()

        # Time slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Time:"))

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(100)
        self.time_slider.setValue(0)
        self.time_slider.valueChanged.connect(self.update_plot_time)

        self.time_label = QLabel("0.0 - 5.0 s")

        slider_layout.addWidget(self.time_slider)
        slider_layout.addWidget(self.time_label)

        # Toggle button for trace display
        toggle_layout = QHBoxLayout()
        toggle_layout.addWidget(QLabel("Display:"))

        self.trace_toggle_btn = QPushButton("Raw & Processed")
        self.trace_toggle_btn.clicked.connect(self.toggle_trace_display)
        self.current_display_mode = "both"

        toggle_layout.addWidget(self.trace_toggle_btn)

        # Probe type label
        self.probe_label = QLabel(f"Probe: {self.probe_type}")
        toggle_layout.addWidget(self.probe_label)
        toggle_layout.addStretch()

        controls_layout.addLayout(slider_layout)
        controls_layout.addLayout(toggle_layout)

        plot_layout.addLayout(controls_layout)

        # Right panel with sorting buttons
        right_layout = QVBoxLayout()

        
        # Sorter selection button
        self.select_sorter_btn = QPushButton("Select Sorter")
        self.select_sorter_btn.clicked.connect(self.select_sorter)
        right_layout.addWidget(self.select_sorter_btn)

        # Sort parameter edit button
        self.edit_params_btn = QPushButton("Edit Parameters")
        self.edit_params_btn.clicked.connect(self.edit_sorter_parameters)
        right_layout.addWidget(self.edit_params_btn)

        # Sort button
        self.sort_btn = QPushButton("Sort")
        self.sort_btn.clicked.connect(self.start_sorting)
        right_layout.addWidget(self.sort_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        # Save EPOCs button
        self.save_epocs_btn = QPushButton("Save EPOCs")
        self.save_epocs_btn.clicked.connect(self.save_epocs)
        right_layout.addWidget(self.save_epocs_btn)

        # Assess Quiality Button
        self.assess_quality_btn = QPushButton("Assess Unit Quality")
        self.assess_quality_btn.clicked.connect(self.assess_unit_quality)
        right_layout.addWidget(self.assess_quality_btn)

        

        # Next File Button
        self.finish_batch_btn = QPushButton("Finish & Next")
        self.finish_batch_btn.clicked.connect(self.finish_current_and_next)
        self.finish_batch_btn.setVisible(False)  # Hidden by default
        right_layout.addWidget(self.finish_batch_btn)


        # Add this after the assess_quality_btn in the right_layout section
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.open_analysis_gui)
        right_layout.addWidget(self.analyze_btn)

        # Also set the button height with the other button height settings
        



        right_layout.addStretch()

        top_layout.addLayout(left_layout, 1)
        top_layout.addLayout(plot_layout, 4)
        top_layout.addLayout(right_layout, 1)

        main_layout.addLayout(top_layout, 3)

        # Bottom section with log and status
        bottom_layout = QHBoxLayout()

        # Log text box
        log_layout = QVBoxLayout()
        log_layout.addWidget(QLabel("Processing Log:"))

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        # Status display
        status_layout = QVBoxLayout()
        status_layout.addWidget(QLabel("Status:"))

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("border: 2px solid gray; padding: 20px; background-color: white;")
        self.status_label.setMaximumHeight(200)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.status_label.setFont(font)
        status_layout.addWidget(self.status_label)

        bottom_layout.addLayout(log_layout, 1)
        bottom_layout.addLayout(status_layout, 1)

        main_layout.addLayout(bottom_layout, 1)

        # Set button sizes
        button_height = 40
        self.manual_select_btn.setFixedHeight(button_height)
        self.load_list_btn.setFixedHeight(button_height)
        self.probe_config_btn.setFixedHeight(button_height)
        self.select_sorter_btn.setFixedHeight(button_height)
        self.edit_params_btn.setFixedHeight(button_height) 
        self.sort_btn.setFixedHeight(button_height)
        self.save_epocs_btn.setFixedHeight(button_height)
        self.assess_quality_btn.setFixedHeight(button_height)
        self.analyze_btn.setFixedHeight(button_height)
        self.trace_toggle_btn.setFixedHeight(30)

        # Initialize probe geometry display
        self.update_probe_geometry()

    def start_sorting(self):
        if not self.recordings:
            self.add_log_message("No recordings loaded. Please load data first.")
            return

        if not self.selected_sorter:
            self.add_log_message("Please select a sorter first.")
            return

        # If no parameters set, get defaults
        if not self.sorter_params:
            try:
                self.sorter_params = si.get_default_sorter_params(self.selected_sorter)
                self.add_log_message(f"Using default parameters for {self.selected_sorter}")
            except Exception as e:
                self.add_log_message(f"Error getting default parameters: {str(e)}")
                self.sorter_params = {}

        self.add_log_message(f"\n=== Starting spike sorting with {self.selected_sorter} ===")
        self.add_log_message(f"Parameters: {self.sorter_params}")

        # Disable sort button and show progress
        self.sort_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Create and start sorting worker
        self.current_sorting_worker = SortingWorker(
            self.recordings,
            self.raw_recordings, 
            self.tdt_data_objects,
            self.file_paths,
            self.selected_sorter,
            self.sorter_params,  # Pass user-defined parameters
            None  # output_base_dir not needed since we use TDT folders
        )

        self.current_sorting_worker.finished.connect(self.on_sorting_finished)
        self.current_sorting_worker.log_message.connect(self.add_log_message)
        self.current_sorting_worker.progress.connect(self.progress_bar.setValue)
        self.current_sorting_worker.start()


    def on_sorting_finished(self, success, message):
        self.sort_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        if success:
            self.sorting_results = self.current_sorting_worker.sorting_results
            self.set_status_success(message)
            self.add_log_message(f"\n=== Sorting completed successfully ===")
            self.add_log_message(f"Results saved in spike_sort folders within each TDT directory")
        else:
            self.set_status_failed(f"Sorting failed: {message}")

        self.current_sorting_worker = None

    def select_sorter(self):
        dialog = SorterSelectionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.selected_sorter = dialog.get_selected_sorter()
            self.add_log_message(f"Selected spike sorter: {self.selected_sorter}")

            # Immediately show parameter dialog
            self.edit_sorter_parameters()
    def edit_sorter_parameters(self):
        """Open parameter editor for the selected sorter"""
        if not self.selected_sorter:
            self.add_log_message("Please select a sorter first")
            return

        try:
            # Show parameter dialog
            param_dialog = SorterParameterDialog(self.selected_sorter, self)
            if param_dialog.exec_() == QDialog.Accepted:
                self.sorter_params = param_dialog.get_parameters()
                self.add_log_message(f"Parameters updated for {self.selected_sorter}")
                self.add_log_message(f"Current parameters: {self.sorter_params}")
            else:
                # If cancelled, use defaults
                self.sorter_params = si.get_default_sorter_params(self.selected_sorter)
                self.add_log_message(f"Using default parameters for {self.selected_sorter}")

        except Exception as e:
            self.add_log_message(f"Error setting up parameters: {str(e)}")
            self.sorter_params = {}



    def get_available_epocs(self, tdt_data):
        """Extract available EPOC names from TDT data"""
        available_epocs = []
        if hasattr(tdt_data, 'epocs'):
            for epoc_name in dir(tdt_data.epocs):
                if not epoc_name.startswith('_'):
                    epoc_obj = getattr(tdt_data.epocs, epoc_name)
                    if hasattr(epoc_obj, 'data') and hasattr(epoc_obj, 'onset'):
                        available_epocs.append(epoc_name)
        return available_epocs

    def save_epocs(self):
        if not self.tdt_data_objects:
            self.add_log_message("No TDT data loaded. Please load data first.")
            return

        # Get available EPOCs from first recording
        available_epocs = self.get_available_epocs(self.tdt_data_objects[0])

        if not available_epocs:
            self.add_log_message("No EPOCs found in the loaded data.")
            return

        # Show EPOC selection dialog
        dialog = EPOCSelectionDialog(available_epocs, self)
        if dialog.exec_() != QDialog.Accepted:
            return

        selected_epocs = dialog.get_selected_epocs()
        if not selected_epocs:
            self.add_log_message("No EPOCs selected.")
            return

        # Save EPOCs to spike_sort folders
        try:
            for recording_idx, (tdt_data, file_path) in enumerate(zip(self.tdt_data_objects, self.file_paths)):
                spike_sort_dir = Path(file_path) / "spike_sort"
                spike_sort_dir.mkdir(exist_ok=True)

                epoc_data = []
                self.add_log_message(f"Processing EPOCs from recording {recording_idx + 1}...")

                for epoc_name in selected_epocs:
                    epoc_obj = getattr(tdt_data.epocs, epoc_name)
                    epoc_values = epoc_obj.data
                    epoc_onsets = epoc_obj.onset

                    for i in range(len(epoc_values)):
                        row = {
                            'epoc_name': epoc_name,
                            'event_index': i,
                            'data_value': epoc_values[i],
                            'onset_time': epoc_onsets[i]
                        }
                        epoc_data.append(row)

                # Save to CSV in spike_sort folder
                df = pd.DataFrame(epoc_data)
                epoc_path = spike_sort_dir / "epocs.csv"
                df.to_csv(epoc_path, index=False)
                self.add_log_message(f"EPOCs saved to: {epoc_path}")

            self.add_log_message("All EPOCs saved successfully")

        except Exception as e:
            self.add_log_message(f"Error saving EPOCs: {str(e)}")

    def assess_unit_quality(self):
        """Open unit quality assessment dialog"""
        if not self.sorting_results:
            self.add_log_message("No sorting results available. Please run spike sorting first.")
            return

        # Let user select which recording to assess
        if len(self.sorting_results) > 1:
            # Create simple selection dialog
            from qtpy.QtWidgets import QInputDialog
            items = [f"Recording {i+1}: {Path(result['file_path']).name}" for i, result in enumerate(self.sorting_results)]
            item, ok = QInputDialog.getItem(self, "Select Recording", "Choose recording to assess:", items, 0, False)
            if not ok:
                return
            recording_idx = items.index(item)
        else:
            recording_idx = 0

        # Get the reports directory
        result = self.sorting_results[recording_idx]
        reports_dir = Path(result['output_dir']) / "spikeinterface_report"

        if not reports_dir.exists():
            self.add_log_message(f"No reports found at {reports_dir}. Please generate unit reports first.")
            return

        # Check if units folder exists
        units_dir = reports_dir / "units"
        if not units_dir.exists():
            self.add_log_message(f"No units folder found at {units_dir}. Please generate unit reports first.")
            return

        # Open quality assessment dialog
        self.add_log_message(f"Opening unit quality assessment for {Path(result['file_path']).name}")
        dialog = UnitQualityDialog(reports_dir, self)
        if dialog.exec_() == QDialog.Accepted:
            quality_data = dialog.get_quality_data()
            self.add_log_message(f"Unit quality assessment completed. Rated {len(quality_data)} units.")
            self.add_log_message(f"Results saved to unit_quality.csv in {result['output_dir']}")

    def update_probe_geometry(self):
        """Create and display the current probe geometry"""
        if self.probe_type == "Linear 16-channel":
            probe = self.create_linear_probe()
        elif self.probe_type == "Tetrode 32-channel":
            probe = self.create_tetrode_probe()
        else:
            probe = self.create_linear_probe()  # Default

        self.current_probe = probe
        self.probe_geometry_widget.update_probe_plot(probe)

    def create_linear_probe(self):
        from probeinterface import Probe
        n = 16
        positions = np.zeros((n, 2))
        for i in range(n):
            x = 0
            y = i * 100
            positions[i] = x, y

        probe = Probe(ndim=2, si_units='um')
        probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 5})
        return probe

    def create_tetrode_probe(self):
        from probeinterface import Probe

        probe = Probe(ndim=2, si_units='um')

        # Tetrode bundle positions (2 rows of 4, 200um separation)
        bundle_positions = []
        for row in range(2):
            for col in range(4):
                x = col * 200  # 200um between bundles
                y = row * 200  # 200um between rows
                bundle_positions.append((x, y))

        # Generate electrode positions (4 electrodes per tetrode in square formation)
        positions = []
        contact_ids = []
        shank_ids = []

        tetrode_offset = 12.5  # 25um tetrode tip separation / 2

        for bundle_idx, (bx, by) in enumerate(bundle_positions):
            # 4 electrodes per tetrode in square pattern
            tetrode_positions = [
                (bx - tetrode_offset, by - tetrode_offset),  # bottom-left
                (bx + tetrode_offset, by - tetrode_offset),  # bottom-right
                (bx - tetrode_offset, by + tetrode_offset),  # top-left
                (bx + tetrode_offset, by + tetrode_offset)   # top-right
            ]

            for i, (x, y) in enumerate(tetrode_positions):
                positions.append([x, y])
                contact_ids.append(bundle_idx * 4 + i)
                shank_ids.append(bundle_idx)

        probe.set_contacts(positions=np.array(positions), 
                          contact_ids=np.array(contact_ids),
                          shank_ids=np.array(shank_ids),
                          shapes='circle',
                          shape_params={'radius':5})
        return probe

    def configure_probe(self):
        dialog = ProbeConfigDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.probe_type = dialog.get_selected_probe()
            self.probe_label.setText(f"Probe: {self.probe_type}")
            self.add_log_message(f"Probe configuration changed to: {self.probe_type}")
            self.update_probe_geometry()

    def toggle_trace_display(self):
        # Cycle through display modes: both -> raw -> processed -> both
        if self.current_display_mode == "both":
            self.current_display_mode = "raw"
            self.trace_toggle_btn.setText("Raw Only")
        elif self.current_display_mode == "raw":
            self.current_display_mode = "processed"
            self.trace_toggle_btn.setText("Processed Only")
        else:  # processed
            self.current_display_mode = "both"
            self.trace_toggle_btn.setText("Raw & Processed")

        # Update plot canvas display mode
        self.plot_canvas.set_display_mode(self.current_display_mode)

        # Refresh the plot
        self.update_plot()

    def manual_select_files(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.status_label.setText("Loading...")
            self.status_label.setStyleSheet("border: 2px solid gray; padding: 20px; background-color: white; color: blue;")
            self.load_data([directory])

    def load_file_list(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            try:
                # Read CSV file
                df = pd.read_csv(file_path)

                # Assume first column contains the paths
                if len(df.columns) == 0:
                    self.set_status_failed("CSV file is empty")
                    return

                # Get paths from first column
                self.batch_file_list = df.iloc[:, 0].dropna().tolist()

                if not self.batch_file_list:
                    self.set_status_failed("No valid paths found in CSV")
                    return

                self.current_batch_index = 0
                self.is_batch_mode = True

                self.add_log_message(f"Loaded batch list with {len(self.batch_file_list)} files")
                self.add_log_message(f"Files: {self.batch_file_list}")

                # Load first file
                self.load_current_batch_file()

            except Exception as e:
                self.set_status_failed(f"Error reading CSV file: {str(e)}")

    def finish_current_and_next(self):
        """Finish current file processing and load next file in batch"""
        if not self.is_batch_mode:
            return

        self.add_log_message(f"Finished processing file {self.current_batch_index + 1}")

        # Move to next file
        self.current_batch_index += 1

        # Clear current data
        self.recordings = []
        self.raw_recordings = []
        self.tdt_data_objects = []
        self.sorting_results = []

        # Clear plot
        self.plot_canvas.fig.clear()
        self.plot_canvas.draw()

        # Load next file or finish batch
        if self.current_batch_index < len(self.batch_file_list):
            self.load_current_batch_file()
        else:
            self.add_log_message("=== Batch processing completed ===")
            self.is_batch_mode = False
            self.finish_batch_btn.setVisible(False)
            self.set_status_success("Batch processing completed")



    def load_current_batch_file(self):
        """Load the current file in the batch list"""
        if not self.is_batch_mode or self.current_batch_index >= len(self.batch_file_list):
            self.add_log_message("Batch processing completed")
            self.is_batch_mode = False
            return

        current_path = self.batch_file_list[self.current_batch_index]
        self.add_log_message(f"\n=== Loading batch file {self.current_batch_index + 1}/{len(self.batch_file_list)} ===")
        self.add_log_message(f"Current file: {current_path}")

        self.status_label.setText(f"Loading batch file {self.current_batch_index + 1}/{len(self.batch_file_list)}...")
        self.status_label.setStyleSheet("border: 2px solid gray; padding: 20px; background-color: white; color: blue;")

        self.load_data([current_path])



    def load_data(self, file_paths, is_batch=False):
        self.log_text.clear()
        self.file_paths = file_paths
        self.current_loader = DataLoader(file_paths, self.probe_type, is_batch)
        self.current_loader.finished.connect(self.on_data_loaded)
        self.current_loader.log_message.connect(self.add_log_message)
        self.current_loader.start()

    def add_log_message(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def on_data_loaded(self, success, message):
        if success:
            self.recordings = self.current_loader.recordings
            self.raw_recordings = self.current_loader.raw_recordings
            self.tdt_data_objects = self.current_loader.tdt_data_objects
            self.set_status_success(message)
            self.update_time_slider()
            self.update_plot()

            # Show finish button only in batch mode
            if self.is_batch_mode:
                self.finish_batch_btn.setVisible(True)
            else:
                self.finish_batch_btn.setVisible(False)
        else:
            self.set_status_failed(message)
            self.finish_batch_btn.setVisible(False)

        self.current_loader = None

    def open_analysis_gui(self):
        """Open the analysis GUI"""
        analysis_dialog = AnalysisGUI(self)
        analysis_dialog.exec_()



    def update_time_slider(self):
        if self.recordings:
            duration = self.recordings[0].get_num_frames() / self.recordings[0].get_sampling_frequency()
            max_start_time = max(0, duration - 5.0)  # 5 second window
            self.time_slider.setMaximum(int(max_start_time * 10))  # 0.1 second resolution

    def update_plot_time(self):
        if self.recordings:
            start_time = self.time_slider.value() / 10.0  # Convert back to seconds
            end_time = start_time + 5.0
            self.time_label.setText(f"{start_time:.1f} - {end_time:.1f} s")
            self.update_plot()

    def update_plot(self):
        if self.recordings and self.raw_recordings:
            start_time = self.time_slider.value() / 10.0
            self.plot_canvas.plot_recordings(
                self.raw_recordings[self.current_recording_idx],
                self.recordings[self.current_recording_idx],
                start_time
            )

    def set_status_success(self, message="Files Loaded Successfully"):
        self.status_label.setText(message)
        self.status_label.setStyleSheet("border: 2px solid gray; padding: 20px; background-color: white; color: green;")

    def set_status_failed(self, message="File Load Failed"):
        self.status_label.setText(message)
        self.status_label.setStyleSheet("border: 2px solid gray; padding: 20px; background-color: white; color: red;")

def main():
    app = QApplication(sys.argv)
    window = SortingPipelineGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
