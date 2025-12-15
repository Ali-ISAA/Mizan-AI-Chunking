"""
Report generator for chunker and embedder jobs
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class JobReport:
    """Generates and saves job reports"""

    def __init__(self, job_type: str, log_to_file: bool = False):
        """
        Initialize job report

        Parameters:
        -----------
        job_type : str
            Type of job ('chunker' or 'embedder')
        log_to_file : bool
            Whether to save logs to file
        """
        self.job_type = job_type
        self.log_to_file = log_to_file
        self.start_time = datetime.now()
        self.end_time = None

        # Stats
        self.total_files = 0
        self.successful_files = 0
        self.failed_files: List[Dict] = []
        self.total_chunks = 0
        self.total_embeddings = 0

        # Config
        self.config: Dict = {}

        # Logs directory
        self.logs_dir = Path('logs')
        if log_to_file:
            self.logs_dir.mkdir(exist_ok=True)

    def set_config(self, **kwargs):
        """Set job configuration"""
        self.config = kwargs

    def add_success(self, file_path: str, chunks: int = 0, embeddings: int = 0):
        """Record a successful file"""
        self.successful_files += 1
        self.total_chunks += chunks
        self.total_embeddings += embeddings

    def add_failure(self, file_path: str, error: str, stage: str = 'unknown'):
        """Record a failed file"""
        self.failed_files.append({
            'file': str(file_path),
            'error': str(error)[:200],  # Truncate long errors
            'stage': stage,
            'timestamp': datetime.now().isoformat()
        })

    def finalize(self):
        """Finalize the report"""
        self.end_time = datetime.now()
        self.total_files = self.successful_files + len(self.failed_files)

    def get_duration(self) -> str:
        """Get job duration as string"""
        if not self.end_time:
            self.end_time = datetime.now()
        delta = self.end_time - self.start_time
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def get_report_dict(self) -> Dict:
        """Get report as dictionary"""
        return {
            'job_type': self.job_type,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.get_duration(),
            'config': self.config,
            'stats': {
                'total_files': self.total_files,
                'successful': self.successful_files,
                'failed': len(self.failed_files),
                'success_rate': f"{100 * self.successful_files / self.total_files:.1f}%" if self.total_files > 0 else "N/A",
                'total_chunks': self.total_chunks,
                'total_embeddings': self.total_embeddings
            },
            'failed_files': self.failed_files
        }

    def save_report(self) -> Optional[str]:
        """Save report to logs directory"""
        if not self.log_to_file:
            return None

        self.finalize()
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        report_file = self.logs_dir / f"{self.job_type}_report_{timestamp}.json"

        report = self.get_report_dict()

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        return str(report_file)

    def save_failed_files_list(self) -> Optional[str]:
        """Save list of failed files for retry"""
        if not self.log_to_file or not self.failed_files:
            return None

        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        failed_file = self.logs_dir / f"{self.job_type}_failed_{timestamp}.txt"

        with open(failed_file, 'w') as f:
            for item in self.failed_files:
                f.write(f"{item['file']}\n")

        return str(failed_file)

    def print_summary(self):
        """Print summary to console"""
        self.finalize()

        print(f"\n{'='*60}")
        print(f"  Job Report: {self.job_type.upper()}")
        print(f"{'='*60}")
        print(f"  Duration:         {self.get_duration()}")
        print(f"  Total files:      {self.total_files}")
        print(f"  Successful:       {self.successful_files}")
        print(f"  Failed:           {len(self.failed_files)}")
        if self.total_files > 0:
            print(f"  Success rate:     {100 * self.successful_files / self.total_files:.1f}%")
        if self.total_chunks > 0:
            print(f"  Total chunks:     {self.total_chunks}")
        if self.total_embeddings > 0:
            print(f"  Total embeddings: {self.total_embeddings}")

        if self.failed_files:
            print(f"\n  Failed files ({len(self.failed_files)}):")
            for item in self.failed_files[:10]:  # Show first 10
                fname = Path(item['file']).name[:40]
                print(f"    - {fname}... [{item['stage']}]")
            if len(self.failed_files) > 10:
                print(f"    ... and {len(self.failed_files) - 10} more")

        if self.log_to_file:
            report_path = self.save_report()
            failed_path = self.save_failed_files_list()
            print(f"\n  Logs saved to:")
            if report_path:
                print(f"    Report: {report_path}")
            if failed_path:
                print(f"    Failed: {failed_path}")

        print()
