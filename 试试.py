MOD = 10 ** 9 + 7


# Use a bottom-up dynamic programming approach
def check_record(n: int) -> int:
    # dp[i][j][k] will be the count of valid sequences of length i where:
    # j indicates the number of A's in the sequence (0 or 1)
    # k indicates the number of trailing L's (0, 1, or 2)
    dp = [[[0] * 3 for _ in range(2)] for _ in range(n + 1)]

    # Base case initialization
    dp[0][0][0] = 1

    for i in range(1, n + 1):
        for j in range(2):
            for k in range(3):
                # When adding a 'P', the count of trailing L's resets to 0
                dp[i][j][0] = (dp[i][j][0] + dp[i - 1][j][k]) % MOD

                # When adding an 'L', the count of trailing L's increases by 1
                if k < 2:
                    dp[i][j][k + 1] = (dp[i][j][k + 1] + dp[i - 1][j][k]) % MOD

                # When adding an 'A', the count of A's increases to 1 and trailing L's reset to 0
                if j == 0:
                    dp[i][1][0] = (dp[i][1][0] + dp[i - 1][j][k]) % MOD

    # Sum up all the valid sequences of length n
    result = 0
    for j in range(2):
        for k in range(3):
            result = (result + dp[n][j][k]) % MOD

    return result


# Example usage:
# n = 2 would be input by the user, but here is an example call to the function
n = 2
print(check_record(n))
