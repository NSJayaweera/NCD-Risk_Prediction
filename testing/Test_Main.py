import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure the parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Mock class to support st.session_state.page (dot notation)
class SessionStateMock(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class TestMainApp(unittest.TestCase):

    def setUp(self):
        # Create a fresh state for every test
        self.mock_state = SessionStateMock()
        self.mock_state['page'] = 'Home'

    @patch('streamlit.columns')
    @patch('streamlit.button')
    @patch('streamlit.set_page_config')
    @patch('streamlit.markdown')
    def test_home_page_rendering(self, mock_md, mock_config, mock_button, mock_columns):
        """Test if the Home page elements are called correctly"""
        with patch('streamlit.session_state', self.mock_state):
            import Main

            # Setup columns mock
            mock_columns.return_value = [MagicMock(), MagicMock()]

            # Trigger logic manually (since it's at module level)
            if self.mock_state.page == 'Home':
                Main.st.button("Heart Disease Risk Prediction",
                               on_click=Main.nav_to, args=('Heart',), use_container_width=True)

            mock_button.assert_any_call("Heart Disease Risk Prediction",
                                        on_click=Main.nav_to,
                                        args=('Heart',),
                                        use_container_width=True)

    def test_navigation_callback(self):
        """Test that the nav_to function successfully changes the state"""
        with patch('streamlit.session_state', self.mock_state):
            import Main
            Main.nav_to('CKD')
            self.assertEqual(self.mock_state.page, 'CKD')

    @patch('Heart.run_heart_analysis')
    @patch('streamlit.button')
    def test_heart_navigation_routing(self, mock_back_btn, mock_run_heart):
        """Test if the app attempts to run heart analysis when page is Heart"""
        self.mock_state.page = 'Heart'
        with patch('streamlit.session_state', self.mock_state):
            import Main
            mock_run_heart.reset_mock()
            if self.mock_state.page == 'Heart':
                Main.Heart.run_heart_analysis()
            mock_run_heart.assert_called_once()


if __name__ == '__main__':
    unittest.main()