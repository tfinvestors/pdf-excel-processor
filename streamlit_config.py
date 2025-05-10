import os
import sys
import locale


def configure_streamlit_environment():
    """Configure environment for Streamlit deployment"""

    # Force UTF-8 encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    os.environ['LANG'] = 'en_US.UTF-8'

    # Configure stdout/stderr
    if sys.platform.startswith('win'):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

    # Disable problematic libraries in Streamlit Cloud
    if 'STREAMLIT_SHARING' in os.environ:
        os.environ['DISABLE_CAMELOT'] = '1'

    return True