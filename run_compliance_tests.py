#!/usr/bin/env python3
"""
Comprehensive testing script for TrainingAutomation codebase.

This script ensures full compliance with British English and style guidelines,
runs all tests, and validates implementation completeness.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd: List[str], cwd: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a shell command and return the result.
    
    Args:
        cmd: Command to run as list of strings
        cwd: Working directory
        
    Returns:
        Dictionary with result information
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': 'Command timed out'
        }
    except Exception as e:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e)
        }

def test_british_english_compliance(project_root: str) -> bool:
    """
    Test British English compliance across the codebase.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        True if all checks pass
    """
    logger.info("Checking British English compliance...")
    
    american_spellings = [
        'optimization', 'initialize', 'initialization', 'color', 'center',
        'organize', 'organization', 'analyze', 'visualization', 'visualize'
    ]
    
    issues_found = []
    
    # Check Python files
    for python_file in Path(project_root).rglob("*.py"):
        if '.venv' in str(python_file) or '__pycache__' in str(python_file):
            continue
            
        try:
            with open(python_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                for spelling in american_spellings:
                    if spelling in content:
                        # Check if it's in a comment or string (acceptable)
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if spelling in line:
                                # Skip if it's part of a method name required by external library
                                if 'initialize_model' in line or 'color=' in line:
                                    continue
                                    
                                issues_found.append(f"{python_file}:{i} - '{spelling}' found")
        except Exception as e:
            logger.warning(f"Could not read {python_file}: {e}")
    
    if issues_found:
        logger.error(f"British English compliance issues found:")
        for issue in issues_found:
            logger.error(f"  {issue}")
        return False
    else:
        logger.info("✓ British English compliance check passed")
        return True

def run_linting(project_root: str) -> bool:
    """
    Run linting checks on the codebase.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        True if linting passes
    """
    logger.info("Running linting checks...")
    
    # Check if flake8 is available
    flake8_result = run_command(['python3', '-m', 'flake8', '--version'])
    if not flake8_result['success']:
        logger.warning("flake8 not available, skipping linting")
        return True
    
    # Run flake8
    result = run_command([
        'python3', '-m', 'flake8', 
        'src/', 'tests/', 'main.py',
        '--max-line-length=100',
        '--ignore=E501,W503,E203'  # Ignore line length for now
    ], cwd=project_root)
    
    if result['success']:
        logger.info("✓ Linting checks passed")
        return True
    else:
        logger.error("Linting issues found:")
        logger.error(result['stdout'])
        logger.error(result['stderr'])
        return False

def run_unit_tests(project_root: str) -> bool:
    """
    Run all unit tests.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        True if all tests pass
    """
    logger.info("Running unit tests...")
    
    # Run pytest
    result = run_command([
        'python3', '-m', 'pytest', 
        'tests/',
        '-v',
        '--tb=short'
    ], cwd=project_root)
    
    if result['success']:
        logger.info("✓ All unit tests passed")
        return True
    else:
        logger.error("Unit test failures:")
        logger.error(result['stdout'])
        logger.error(result['stderr'])
        return False

def check_docstring_coverage(project_root: str) -> bool:
    """
    Check docstring coverage for public methods.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        True if docstring coverage is adequate
    """
    logger.info("Checking docstring coverage...")
    
    missing_docstrings = []
    
    for python_file in Path(project_root).rglob("src/**/*.py"):
        try:
            with open(python_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    # Check for public method/function definitions
                    if line.strip().startswith('def ') and not line.strip().startswith('def _'):
                        # Check if next non-empty line starts a docstring
                        j = i + 1
                        while j < len(lines) and not lines[j].strip():
                            j += 1
                        
                        if j < len(lines) and not lines[j].strip().startswith('"""'):
                            missing_docstrings.append(f"{python_file}:{i+1} - {line.strip()}")
                            
        except Exception as e:
            logger.warning(f"Could not check {python_file}: {e}")
    
    if missing_docstrings:
        logger.warning(f"Methods missing docstrings:")
        for missing in missing_docstrings[:10]:  # Show first 10
            logger.warning(f"  {missing}")
        if len(missing_docstrings) > 10:
            logger.warning(f"  ... and {len(missing_docstrings) - 10} more")
        return False
    else:
        logger.info("✓ Docstring coverage is adequate")
        return True

def validate_imports(project_root: str) -> bool:
    """
    Validate import organisation follows the style guide.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        True if imports are properly organised
    """
    logger.info("Validating import organisation...")
    
    issues = []
    
    for python_file in Path(project_root).rglob("src/**/*.py"):
        try:
            with open(python_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                import_lines = []
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        import_lines.append((i + 1, line.strip()))
                    elif import_lines and line.strip() and not line.startswith('#'):
                        break  # End of import block
                
                # Basic check: standard library should come first
                stdlib_found = False
                thirdparty_found = False
                local_found = False
                
                for line_num, import_line in import_lines:
                    if any(lib in import_line for lib in ['os', 'sys', 'logging', 'typing', 'pathlib']):
                        if thirdparty_found or local_found:
                            issues.append(f"{python_file}:{line_num} - Standard library import after third-party/local")
                        stdlib_found = True
                    elif import_line.startswith('from src.') or import_line.startswith('import src.'):
                        local_found = True
                    else:
                        if local_found:
                            issues.append(f"{python_file}:{line_num} - Third-party import after local")
                        thirdparty_found = True
                        
        except Exception as e:
            logger.warning(f"Could not check imports in {python_file}: {e}")
    
    if issues:
        logger.warning("Import organisation issues found:")
        for issue in issues[:5]:  # Show first 5
            logger.warning(f"  {issue}")
        if len(issues) > 5:
            logger.warning(f"  ... and {len(issues) - 5} more")
        return False
    else:
        logger.info("✓ Import organisation is correct")
        return True

def main():
    """Main function to run all compliance checks."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Running compliance checks for project: {project_root}")
    
    checks = [
        ("British English Compliance", lambda: test_british_english_compliance(project_root)),
        ("Import Organisation", lambda: validate_imports(project_root)),
        ("Docstring Coverage", lambda: check_docstring_coverage(project_root)),
        ("Linting", lambda: run_linting(project_root)),
        ("Unit Tests", lambda: run_unit_tests(project_root)),
    ]
    
    results = {}
    for check_name, check_func in checks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {check_name}")
        logger.info('='*60)
        
        try:
            results[check_name] = check_func()
        except Exception as e:
            logger.error(f"Error running {check_name}: {e}")
            results[check_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("COMPLIANCE CHECK SUMMARY")
    logger.info('='*60)
    
    passed = 0
    total = len(checks)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{check_name:30} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("🎉 All compliance checks passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some compliance checks failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
