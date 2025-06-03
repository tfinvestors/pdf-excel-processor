import os
import sys
import locale


def configure_streamlit_environment():
    """Configure environment for Streamlit deployment"""

    # Force UTF-8 encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    # Set locale if possible
    try:
        if sys.platform.startswith('win'):
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        else:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except locale.Error:
        pass  # Ignore if locale setting fails

    # Configure stdout/stderr for better encoding handling
    if sys.platform.startswith('win'):
        try:
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        except:
            pass  # Fall back to default if this fails

    # Disable problematic libraries in Streamlit Cloud
    if 'STREAMLIT_SHARING' in os.environ:
        os.environ['DISABLE_CAMELOT'] = '1'

    return True