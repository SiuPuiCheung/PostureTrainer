"""Report generation utilities for pose evaluation."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List
from ..utils.config_loader import Config


class ReportGenerator:
    """Generates PDF reports with joint angle plots and statistics."""
    
    def __init__(self, config: Config):
        """
        Initialize report generator with configuration.
        
        Args:
            config: Configuration object containing report settings.
        """
        self.config = config
        self.report_config = config.report_config
    
    def generate_report(self, 
                       joint_angles_df: pd.DataFrame, 
                       frame_rate: float, 
                       analysis_choice: int, 
                       timestamp: str,
                       output_dir: str = "output") -> None:
        """
        Generate a PDF report containing plots of joint angles and statistical summaries.
        
        Args:
            joint_angles_df: DataFrame containing joint angles data.
            frame_rate: Frame rate of the video.
            analysis_choice: Index of the chosen analysis type.
            timestamp: Timestamp string for file naming.
            output_dir: Output directory path.
        """
        # Get body labels for the chosen analysis
        body_labels = self.config.get_body_labels_by_index(analysis_choice)
        
        # Get joints to analyze
        joints = joint_angles_df.columns.tolist()
        if analysis_choice == 0 or analysis_choice == 3:
            joints = joints[:-2]
        elif analysis_choice == 3:
            joints = joints[:-1]

        # Add time column
        joint_angles_df['Time'] = joint_angles_df.index / frame_rate

        # Check for matching joint count
        if len(body_labels) != len(joints):
            # Create simple "No Posture detected" report
            fig = plt.figure(figsize=(10, 6))
            plt.suptitle(self.report_config['title'], 
                        fontsize=self.report_config['title_fontsize'])
            plt.text(0.5, 0.5, 'No Posture detected.', fontsize=14, ha='center')
            plt.axis('off')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"Report_{timestamp}.pdf"))
            plt.close(fig)
            return

        # Prepare figure for detailed report
        num_rows = ((len(joints) + 1) // 2) + 5
        fig_width = self.report_config['figure_width']
        fig_height = num_rows * self.report_config['base_height_per_row']
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.suptitle(self.report_config['title'], 
                    fontsize=self.report_config['title_fontsize'], y=1)
        gs = GridSpec(num_rows, 4, figure=fig, width_ratios=[3, 1, 3, 1])

        # Loop through each joint to plot
        for i, joint in enumerate(joints):
            row, col = i // 2, (i % 2) * 2
            
            # Plot time series
            ax_plot = fig.add_subplot(gs[row, col])
            ax_plot.plot(joint_angles_df['Time'], joint_angles_df[joint], label=joint)
            ax_plot.set_title(body_labels[i])
            ax_plot.set_xlabel('Time (s)')
            ax_plot.set_ylabel('Angle (degrees)')
            ax_plot.grid(True)

            # Calculate statistics
            stats = {
                'Average': round(np.mean(joint_angles_df[joint]), 2),
                'Max': round(np.max(joint_angles_df[joint]), 2),
                'Min': round(np.min(joint_angles_df[joint]), 2)
            }
            
            # Create statistics table
            ax_table = fig.add_subplot(gs[row, col + 1])
            ax_table.axis('off')
            table = ax_table.table(
                cellText=[[k, f'{v:.2f}'] for k, v in stats.items()],
                colLabels=['Stat', 'Value'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(True)
            table.scale(1, 1.5)

        plt.subplots_adjust(hspace=1)
        fig.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"Report_{timestamp}.pdf"))
        plt.close(fig)
