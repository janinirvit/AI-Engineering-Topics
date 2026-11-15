"""
Unit tests for chain type functionality in NewsArticleSummarizer.
These tests verify that the chain type parameter works correctly without requiring API calls.
"""

import sys
from unittest.mock import MagicMock, patch


def test_chain_type_validation():
    """Test that chain types are properly validated."""
    print("\n" + "=" * 70)
    print("TEST 1: Chain Type Validation")
    print("=" * 70)

    # Mock the imports to avoid dependency issues
    with patch.dict(
        sys.modules,
        {
            "langchain_classic.chains.summarize": MagicMock(),
            "langchain_core.documents": MagicMock(),
            "langchain_core.prompts": MagicMock(),
            "langchain_openai": MagicMock(),
            "langchain_text_splitters": MagicMock(),
            "newspaper": MagicMock(),
            "dotenv": MagicMock(),
        },
    ):
        # Mock the ChatOpenAI class
        with patch("news_summarizer.ChatOpenAI"):
            from news_summarizer import NewsArticleSummarizer

            # Test valid chain types
            valid_chains = ["stuff", "map_reduce", "refine"]
            print("\nTesting valid chain types:")

            for chain in valid_chains:
                try:
                    # Create instance to call method
                    summarizer = NewsArticleSummarizer(api_key="test-key")
                    result = summarizer._validate_chain_type(chain)
                    assert result == chain.lower(), f"Expected {chain.lower()}, got {result}"
                    print(f"  [PASS] '{chain}' - Valid")
                except Exception as e:
                    print(f"  [FAIL] '{chain}' - Failed: {e}")
                    return False

            # Test case insensitivity
            print("\nTesting case insensitivity:")
            test_cases = ["STUFF", "Map_Reduce", "REFINE"]
            for chain in test_cases:
                try:
                    summarizer = NewsArticleSummarizer(api_key="test-key")
                    result = summarizer._validate_chain_type(chain)
                    assert result == chain.lower(), f"Expected {chain.lower()}, got {result}"
                    print(f"  [PASS] '{chain}' - Converted to '{result}'")
                except Exception as e:
                    print(f"  [FAIL] '{chain}' - Failed: {e}")
                    return False

            # Test invalid chain type
            print("\nTesting invalid chain type rejection:")
            try:
                summarizer = NewsArticleSummarizer(api_key="test-key")
                summarizer._validate_chain_type("invalid_chain")
                print("  [FAIL] Should have raised ValueError")
                return False
            except ValueError as e:
                error_msg = str(e)
                assert "invalid_chain" in error_msg, "Error should mention the invalid chain"
                assert "stuff" in error_msg, "Error should list valid options"
                print("  [PASS] Correctly rejected 'invalid_chain'")
                print(f"         Error: {error_msg[:80]}...")

            return True


def test_chain_type_parameter():
    """Test that chain_type parameter is properly stored."""
    print("\n" + "=" * 70)
    print("TEST 2: Chain Type Parameter Storage")
    print("=" * 70)

    with patch.dict(
        sys.modules,
        {
            "langchain_classic.chains.summarize": MagicMock(),
            "langchain_core.documents": MagicMock(),
            "langchain_core.prompts": MagicMock(),
            "langchain_openai": MagicMock(),
            "langchain_text_splitters": MagicMock(),
            "newspaper": MagicMock(),
            "dotenv": MagicMock(),
        },
    ):
        # Mock the ChatOpenAI class
        with patch("news_summarizer.ChatOpenAI"):
            from news_summarizer import NewsArticleSummarizer

            # Test with different chain types
            test_cases = [
                ("stuff", "stuff"),
                ("map_reduce", "map_reduce"),
                ("REFINE", "refine"),
            ]

            print("\nTesting chain type parameter storage:")
            for input_chain, expected_chain in test_cases:
                try:
                    summarizer = NewsArticleSummarizer(api_key="test-key", chain_type=input_chain)
                    assert summarizer.chain_type == expected_chain, (
                        f"Expected {expected_chain}, got {summarizer.chain_type}"
                    )
                    print(f"  [PASS] Input '{input_chain}' -> Stored as '{summarizer.chain_type}'")
                except Exception as e:
                    print(f"  [FAIL] Failed for '{input_chain}': {e}")
                    return False

            # Test default chain type
            print("\nTesting default chain type:")
            try:
                summarizer = NewsArticleSummarizer(api_key="test-key")
                assert summarizer.chain_type == "map_reduce", (
                    f"Default should be 'map_reduce', got '{summarizer.chain_type}'"
                )
                print("  [PASS] Default chain type is 'map_reduce'")
            except Exception as e:
                print(f"  [FAIL] Failed: {e}")
                return False

            return True


def test_valid_chain_types_constant():
    """Test that VALID_CHAIN_TYPES constant is defined correctly."""
    print("\n" + "=" * 70)
    print("TEST 3: VALID_CHAIN_TYPES Constant")
    print("=" * 70)

    with patch.dict(
        sys.modules,
        {
            "langchain_classic.chains.summarize": MagicMock(),
            "langchain_core.documents": MagicMock(),
            "langchain_core.prompts": MagicMock(),
            "langchain_openai": MagicMock(),
            "langchain_text_splitters": MagicMock(),
            "newspaper": MagicMock(),
            "dotenv": MagicMock(),
        },
    ):
        from news_summarizer import NewsArticleSummarizer

        print("\nChecking VALID_CHAIN_TYPES constant:")
        expected_chains = ["stuff", "map_reduce", "refine"]

        assert hasattr(NewsArticleSummarizer, "VALID_CHAIN_TYPES"), "VALID_CHAIN_TYPES constant not found"
        print(f"  [PASS] Constant exists: {NewsArticleSummarizer.VALID_CHAIN_TYPES}")

        assert NewsArticleSummarizer.VALID_CHAIN_TYPES == expected_chains, (
            f"Expected {expected_chains}, got {NewsArticleSummarizer.VALID_CHAIN_TYPES}"
        )
        print(f"  [PASS] Contains all expected chain types: {len(expected_chains)}")

        return True


def test_return_dict_includes_chain_type():
    """Test that return dict includes chain_type field."""
    print("\n" + "=" * 70)
    print("TEST 4: Return Dictionary Includes chain_type")
    print("=" * 70)

    print("\nChecking news_summarizer.py source code:")
    with open("news_summarizer.py", "r") as f:
        content = f.read()

        # Check for chain_type in return dict
        if '"chain_type": self.chain_type' in content:
            print("  [PASS] Return dictionary includes chain_type field")
            return True
        else:
            print("  [FAIL] chain_type not found in return dictionary")
            return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CHAIN TYPE FUNCTIONALITY TESTS")
    print("=" * 70)

    tests = [
        test_chain_type_validation,
        test_chain_type_parameter,
        test_valid_chain_types_constant,
        test_return_dict_includes_chain_type,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Test failed with exception: {e}")
            import traceback

            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r)
    total = len(results)

    if passed == total:
        print(f"\nALL {total} TESTS PASSED!")
        print("\nChain type selector is ready for use!")
        return 0
    else:
        print(f"\n{total - passed} of {total} tests FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
