import argparse
import json
import time

import jax
import numpy as np
import optax

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer
import transformers
from smart_open import open

from mesh_transformer.util import clip_by_global_norm

from google.cloud import storage


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")

    args = parser.parse_args()
    return args


def save(bucket, path, filename,content):

    # create metadata file
    with open(f"gs://{bucket}/{path}/{filename}", "w") as f:
        f.write(content)

if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]

    params["sampler"] = nucleaus_sample
    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    )

    params["optimizer"] = opt

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    with open(f"gs://{bucket}/{model_dir}/meta.json", "r") as f:
        meta = json.load(f)

    ckpt_step = meta["checkpoints"][-1]
    print(f"using checkpoint {ckpt_step}")

    total_batch = per_replica_batch * jax.device_count() // cores_per_replica
    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        network = CausalTransformer(params)

        start = time.time()
        network.state = read_ckpt(network.state, f"gs://{bucket}/{model_dir}/step_{ckpt_step}/", devices.shape[1])
        print(f"network loaded in {time.time() - start:.06}s")

        local_shards = max(jax.local_device_count() // mesh_shape[1], 1)
        del network.state["opt_state"]
        network.state = network.move_xmap(network.state, np.zeros(local_shards))

        tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

        arr_context=["Andi and Budi were given an assignment to tidy up their bookshelf of n books. Each book is represented by the book title — a string s_i numbered from 1 to n, each with length m. Andi really wants to sort the book lexicographically ascending, while Budi wants to sort it lexicographically descending.\n\nSettling their fight, they decided to combine their idea and sort it asc-desc-endingly, where the odd-indexed characters will be compared ascendingly, and the even-indexed characters will be compared descendingly.\n\nA string a occurs before a string b in asc-desc-ending order if and only if in the first position where a and b differ, the following holds:\n\n  * if it is an odd position, the string a has a letter that appears earlier in the alphabet than the corresponding letter in b; \n  * if it is an even position, the string a has a letter that appears later in the alphabet than the corresponding letter in b. \n\nInput\n\nThe first line contains two integers n and m (1 ≤ n ⋅ m ≤ 10^6).\n\nThe i-th of the next n lines contains a string s_i consisting of m uppercase Latin letters — the book title. The strings are pairwise distinct.\n\nOutput\n\nOutput n integers — the indices of the strings after they are sorted asc-desc-endingly.\n\nExample\n\nInput\n\n\n5 2\nAA\nAB\nBB\nBA\nAZ\n\n\nOutput\n\n\n5 2 1 3 4\n\nNote\n\nThe following illustrates the first example.\n\n<image>",
                     "Mr. Chanek lives in a city represented as a plane. He wants to build an amusement park in the shape of a circle of radius r. The circle must touch the origin (point (0, 0)).\n\nThere are n bird habitats that can be a photo spot for the tourists in the park. The i-th bird habitat is at point p_i = (x_i, y_i). \n\nFind the minimum radius r of a park with at least k bird habitats inside. \n\nA point is considered to be inside the park if and only if the distance between p_i and the center of the park is less than or equal to the radius of the park. Note that the center and the radius of the park do not need to be integers.\n\nIn this problem, it is guaranteed that the given input always has a solution with r ≤ 2 ⋅ 10^5.\n\nInput\n\nThe first line contains two integers n and k (1 ≤ n ≤ 10^5, 1 ≤ k ≤ n) — the number of bird habitats in the city and the number of bird habitats required to be inside the park.\n\nThe i-th of the next n lines contains two integers x_i and y_i (0 ≤ |x_i|, |y_i| ≤ 10^5) — the position of the i-th bird habitat.\n\nOutput\n\nOutput a single real number r denoting the minimum radius of a park with at least k bird habitats inside. It is guaranteed that the given input always has a solution with r ≤ 2 ⋅ 10^5.\n\nYour answer is considered correct if its absolute or relative error does not exceed 10^{-4}.\n\nFormally, let your answer be a, and the jury's answer be b. Your answer is accepted if and only if \\frac{|a - b|}{max{(1, |b|)}} ≤ 10^{-4}.\n\nExamples\n\nInput\n\n\n8 4\n-3 1\n-4 4\n1 5\n2 2\n2 -2\n-2 -4\n-1 -1\n-6 0\n\n\nOutput\n\n\n3.1622776589\n\n\nInput\n\n\n1 1\n0 0\n\n\nOutput\n\n\n0.0000000000\n\nNote\n\nIn the first example, Mr. Chanek can put the center of the park at (-3, -1) with radius √{10} ≈ 3.162. It can be proven this is the minimum r.\n\nThe following illustrates the first example. The blue points represent bird habitats and the red circle represents the amusement park.\n\n<image>",
                     "Mr. Chanek has an integer represented by a string s. Zero or more digits have been erased and are denoted by the character _. There are also zero or more digits marked by the character X, meaning they're the same digit.\n\nMr. Chanek wants to count the number of possible integer s, where s is divisible by 25. Of course, s must not contain any leading zero. He can replace the character _ with any digit. He can also replace the character X with any digit, but it must be the same for every character X.\n\nAs a note, a leading zero is any 0 digit that comes before the first nonzero digit in a number string in positional notation. For example, 0025 has two leading zeroes. An exception is the integer zero, (0 has no leading zero, but 0000 has three leading zeroes).\n\nInput\n\nOne line containing the string s (1 ≤ |s| ≤ 8). The string s consists of the characters 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, _, and X.\n\nOutput\n\nOutput an integer denoting the number of possible integer s.\n\nExamples\n\nInput\n\n\n25\n\n\nOutput\n\n\n1\n\n\nInput\n\n\n_00\n\n\nOutput\n\n\n9\n\n\nInput\n\n\n_XX\n\n\nOutput\n\n\n9\n\n\nInput\n\n\n0\n\n\nOutput\n\n\n1\n\n\nInput\n\n\n0_25\n\n\nOutput\n\n\n0\n\nNote\n\nIn the first example, the only possible s is 25.\n\nIn the second and third example, s ∈ \\{100, 200,300,400,500,600,700,800,900\\}.\n\nIn the fifth example, all possible s will have at least one leading zero.",
                     "Mr. Chanek opened a letter from his fellow, who is currently studying at Singanesia. Here is what it says.\n\nDefine an array b (0 ≤ b_i < k) with n integers. While there exists a pair (i, j) such that b_i ≠ b_j, do the following operation:\n\n  * Randomly pick a number i satisfying 0 ≤ i < n. Note that each number i has a probability of 1/n to be picked. \n  * Randomly Pick a number j satisfying 0 ≤ j < k. \n  * Change the value of b_i to j. It is possible for b_i to be changed to the same value. \n\n\n\nDenote f(b) as the expected number of operations done to b until all elements of b are equal. \n\nYou are given two integers n and k, and an array a (-1 ≤ a_i < k) of n integers. \n\nFor every index i with a_i = -1, replace a_i with a random number j satisfying 0 ≤ j < k. Let c be the number of occurrences of -1 in a. There are k^c possibilites of a after the replacement, each with equal probability of being the final array.\n\nFind the expected value of f(a) modulo 10^9 + 7. \n\nFormally, let M = 10^9 + 7. It can be shown that the answer can be expressed as an irreducible fraction p/q, where p and q are integers and q not ≡ 0 \\pmod{M}. Output the integer equal to p ⋅ q^{-1} mod M. In other words, output such an integer x that 0 ≤ x < M and x ⋅ q ≡ p \\pmod{M}.\n\nAfter reading the letter, Mr. Chanek gave the task to you. Solve it for the sake of their friendship!\n\nInput\n\nThe first line contains two integers n and k (2 ≤ n ≤ 10^5, 2 ≤ k ≤ 10^9). \n\nThe second line contains n integers a_1, a_2, …, a_n (-1 ≤ a_i < k).\n\nOutput\n\nOutput an integer denoting the expected value of f(a) modulo 10^9 + 7.\n\nExamples\n\nInput\n\n\n2 2\n0 1\n\n\nOutput\n\n\n2\n\n\nInput\n\n\n2 2\n0 -1\n\n\nOutput\n\n\n1\n\n\nInput\n\n\n3 3\n0 1 1\n\n\nOutput\n\n\n12\n\n\nInput\n\n\n3 3\n-1 -1 -1\n\n\nOutput\n\n\n11\n\n\nInput\n\n\n10 9\n-1 0 -1 1 1 2 2 3 3 3\n\n\nOutput\n\n\n652419213",
                     "Chanek Jones is back, helping his long-lost relative Indiana Jones, to find a secret treasure in a maze buried below a desert full of illusions.\n\nThe map of the labyrinth forms a tree with n rooms numbered from 1 to n and n - 1 tunnels connecting them such that it is possible to travel between each pair of rooms through several tunnels.\n\nThe i-th room (1 ≤ i ≤ n) has a_i illusion rate. To go from the x-th room to the y-th room, there must exist a tunnel between x and y, and it takes max(|a_x + a_y|, |a_x - a_y|) energy. |z| denotes the absolute value of z.\n\nTo prevent grave robbers, the maze can change the illusion rate of any room in it. Chanek and Indiana would ask q queries.\n\nThere are two types of queries to be done:\n\n  * 1\\ u\\ c — The illusion rate of the x-th room is changed to c (1 ≤ u ≤ n, 0 ≤ |c| ≤ 10^9). \n  * 2\\ u\\ v — Chanek and Indiana ask you the minimum sum of energy needed to take the secret treasure at room v if they are initially at room u (1 ≤ u, v ≤ n). \n\n\n\nHelp them, so you can get a portion of the treasure!\n\nInput\n\nThe first line contains two integers n and q (2 ≤ n ≤ 10^5, 1 ≤ q ≤ 10^5) — the number of rooms in the maze and the number of queries.\n\nThe second line contains n integers a_1, a_2, …, a_n (0 ≤ |a_i| ≤ 10^9) — inital illusion rate of each room.\n\nThe i-th of the next n-1 lines contains two integers s_i and t_i (1 ≤ s_i, t_i ≤ n), meaning there is a tunnel connecting s_i-th room and t_i-th room. The given edges form a tree.\n\nThe next q lines contain the query as described. The given queries are valid.\n\nOutput\n\nFor each type 2 query, output a line containing an integer — the minimum sum of energy needed for Chanek and Indiana to take the secret treasure.\n\nExample\n\nInput\n\n\n6 4\n10 -9 2 -1 4 -6\n1 5\n5 4\n5 6\n6 2\n6 3\n2 1 2\n1 1 -3\n2 1 2\n2 3 3\n\n\nOutput\n\n\n39\n32\n0\n\nNote\n\n<image>\n\nIn the first query, their movement from the 1-st to the 2-nd room is as follows.\n\n  * 1 → 5 — takes max(|10 + 4|, |10 - 4|) = 14 energy. \n  * 5 → 6 — takes max(|4 + (-6)|, |4 - (-6)|) = 10 energy. \n  * 6 → 2 — takes max(|-6 + (-9)|, |-6 - (-9)|) = 15 energy. \n\nIn total, it takes 39 energy.\n\nIn the second query, the illusion rate of the 1-st room changes from 10 to -3.\n\nIn the third query, their movement from the 1-st to the 2-nd room is as follows.\n\n  * 1 → 5 — takes max(|-3 + 4|, |-3 - 4|) = 7 energy. \n  * 5 → 6 — takes max(|4 + (-6)|, |4 - (-6)|) = 10 energy. \n  * 6 → 2 — takes max(|-6 + (-9)|, |-6 - (-9)|) = 15 energy. \n\n\n\nNow, it takes 32 energy.",
                     "Mr. Chanek has a new game called Dropping Balls. Initially, Mr. Chanek has a grid a of size n × m\n\nEach cell (x,y) contains an integer a_{x,y} denoting the direction of how the ball will move.\n\n  * a_{x,y}=1 — the ball will move to the right (the next cell is (x, y + 1)); \n  * a_{x,y}=2 — the ball will move to the bottom (the next cell is (x + 1, y)); \n  * a_{x,y}=3 — the ball will move to the left (the next cell is (x, y - 1)). \n\n\n\nEvery time a ball leaves a cell (x,y), the integer a_{x,y} will change to 2. Mr. Chanek will drop k balls sequentially, each starting from the first row, and on the c_1, c_2, ..., c_k-th (1 ≤ c_i ≤ m) columns.\n\nDetermine in which column each ball will end up in (position of the ball after leaving the grid).\n\nInput\n\nThe first line contains three integers n, m, and k (1 ≤ n, m ≤ 1000, 1 ≤ k ≤ 10^5) — the size of the grid and the number of balls dropped by Mr. Chanek.\n\nThe i-th of the next n lines contains m integers a_{i,1},a_{i,2},…,a_{i,m} (1 ≤ a_{i,j} ≤ 3). It will satisfy a_{i, 1} ≠ 3 and a_{i, m} ≠ 1.\n\nThe next line contains k integers c_1, c_2, …, c_k (1 ≤ c_i ≤ m) — the balls' column positions dropped by Mr. Chanek sequentially.\n\nOutput\n\nOutput k integers — the i-th integer denoting the column where the i-th ball will end.\n\nExamples\n\nInput\n\n\n5 5 3\n1 2 3 3 3\n2 2 2 2 2\n2 2 2 2 2\n2 2 2 2 2\n2 2 2 2 2\n1 2 1\n\n\nOutput\n\n\n2 2 1 \n\n\nInput\n\n\n1 2 2\n1 3\n1 2\n\n\nOutput\n\n\n1 2 \n\nNote\n\nIn the first example, the first ball will drop as follows. Note that the cell (1, 1) will change direction to the bottom direction.\n\n<image>\n\nThe second and third balls will drop as follows. \n\n<image>\n\nAll balls will be dropped from the first row and on the c_1, c_2, ..., c_k-th columns respectively. A ball will stop dropping once it leaves the grid.",
                     "Mr. Chanek gives you a sequence a indexed from 1 to n. Define f(a) as the number of indices where a_i = i. \n\nYou can pick an element from the current sequence and remove it, then concatenate the remaining elements together. For example, if you remove the 3-rd element from the sequence [4, 2, 3, 1], the resulting sequence will be [4, 2, 1]. \n\nYou want to remove some elements from a in order to maximize f(a), using zero or more operations. Find the largest possible f(a).\n\nInput\n\nThe first line contains one integer n (1 ≤ n ≤ 2 ⋅ 10^5) — the initial length of the sequence.\n\nThe second line contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ 2 ⋅ 10^5) — the initial sequence a.\n\nOutput\n\nOutput an integer denoting the largest f(a) that can be obtained by doing zero or more operations.\n\nExamples\n\nInput\n\n\n7\n2 1 4 2 5 3 7\n\n\nOutput\n\n\n3\n\n\nInput\n\n\n4\n4 2 3 1\n\n\nOutput\n\n\n2\n\nNote\n\nIn the first example, f(A) = 3 by doing the following operations.\n\n[2,1,4,2,5,3,7] → [2,1,2,5,3,7] → [1,2,5,3,7] → [1,2,5,3] → [1,2,3]\n\nIn the second example, f(A) = 2 and no additional operation is needed.",
                     "Mr. Chanek's city can be represented as a plane. He wants to build a housing complex in the city.\n\nThere are some telephone poles on the plane, which is represented by a grid a of size (n + 1) × (m + 1). There is a telephone pole at (x, y) if a_{x, y} = 1.\n\nFor each point (x, y), define S(x, y) as the square of the Euclidean distance between the nearest pole and (x, y). Formally, the square of the Euclidean distance between two points (x_1, y_1) and (x_2, y_2) is (x_2 - x_1)^2 + (y_2 - y_1)^2.\n\nTo optimize the building plan, the project supervisor asks you the sum of all S(x, y) for each 0 ≤ x ≤ n and 0 ≤ y ≤ m. Help him by finding the value of ∑_{x=0}^{n} {∑_{y=0}^{m} {S(x, y)}}.\n\nInput\n\nThe first line contains two integers n and m (0 ≤ n, m < 2000) — the size of the grid.\n\nThen (n + 1) lines follow, each containing (m + 1) integers a_{i, j} (0 ≤ a_{i, j} ≤ 1) — the grid denoting the positions of telephone poles in the plane. There is at least one telephone pole in the given grid.\n\nOutput\n\nOutput an integer denoting the value of ∑_{x=0}^{n} {∑_{y=0}^{m} {S(x, y)}}.\n\nExamples\n\nInput\n\n\n2 2\n101\n000\n000\n\n\nOutput\n\n\n18\n\n\nInput\n\n\n5 4\n10010\n00000\n01000\n00001\n00100\n00010\n\n\nOutput\n\n\n36\n\nNote\n\n<image>\n\nIn the first example, the nearest telephone pole for the points (0,0), (1,0), (2,0), (0,1), (1,1), and (2,1) is at (0, 0). While the nearest telephone pole for the points (0, 2), (1,2), and (2,2) is at (0, 2). Thus, ∑_{x=0}^{n} {∑_{y=0}^{m} {S(x, y)}} = (0 + 1 + 4) + (1 + 2 + 5) + (0 + 1 + 4) = 18.",
                     "Casimir has a string s which consists of capital Latin letters 'A', 'B', and 'C' only. Each turn he can choose to do one of the two following actions:\n\n  * he can either erase exactly one letter 'A' and exactly one letter 'B' from arbitrary places of the string (these letters don't have to be adjacent); \n  * or he can erase exactly one letter 'B' and exactly one letter 'C' from arbitrary places in the string (these letters don't have to be adjacent). \n\n\n\nTherefore, each turn the length of the string is decreased exactly by 2. All turns are independent so for each turn, Casimir can choose any of two possible actions.\n\nFor example, with s = \"ABCABC\" he can obtain a string s = \"ACBC\" in one turn (by erasing the first occurrence of 'B' and the second occurrence of 'A'). There are also many other options for a turn aside from this particular example.\n\nFor a given string s determine whether there is a sequence of actions leading to an empty string. In other words, Casimir's goal is to erase all letters from the string. Is there a way to do this?\n\nInput\n\nThe first line contains an integer t (1 ≤ t ≤ 1000) — the number of test cases.\n\nEach test case is described by one string s, for which you need to determine if it can be fully erased by some sequence of turns. The string s consists of capital letters 'A', 'B', 'C' and has a length from 1 to 50 letters, inclusive.\n\nOutput\n\nPrint t lines, each line containing the answer to the corresponding test case. The answer to a test case should be YES if there is a way to fully erase the corresponding string and NO otherwise.\n\nYou may print every letter in any case you want (so, for example, the strings yEs, yes, Yes, and YES will all be recognized as positive answers).\n\nExample\n\nInput\n\n\n6\nABACAB\nABBA\nAC\nABC\nCABCBB\nBCBCBCBCBCBCBCBC\n\n\nOutput\n\n\nNO\nYES\nNO\nNO\nYES\nYES",
                     "The new generation external memory contains an array of integers a[1 … n] = [a_1, a_2, …, a_n].\n\nThis type of memory does not support changing the value of an arbitrary element. Instead, it allows you to cut out any segment of the given array, cyclically shift (rotate) it by any offset and insert it back into the same place.\n\nTechnically, each cyclic shift consists of two consecutive actions: \n\n  1. You may select arbitrary indices l and r (1 ≤ l < r ≤ n) as the boundaries of the segment. \n  2. Then you replace the segment a[l … r] with it's cyclic shift to the left by an arbitrary offset d. The concept of a cyclic shift can be also explained by following relations: the sequence [1, 4, 1, 3] is a cyclic shift of the sequence [3, 1, 4, 1] to the left by the offset 1 and the sequence [4, 1, 3, 1] is a cyclic shift of the sequence [3, 1, 4, 1] to the left by the offset 2. \n\n\n\nFor example, if a = [1, \\color{blue}{3, 2, 8}, 5], then choosing l = 2, r = 4 and d = 2 yields a segment a[2 … 4] = [3, 2, 8]. This segment is then shifted by the offset d = 2 to the left, and you get a segment [8, 3, 2] which then takes the place of of the original elements of the segment. In the end you get a = [1, \\color{blue}{8, 3, 2}, 5].\n\nSort the given array a using no more than n cyclic shifts of any of its segments. Note that you don't need to minimize the number of cyclic shifts. Any method that requires n or less cyclic shifts will be accepted.\n\nInput\n\nThe first line contains an integer t (1 ≤ t ≤ 1000) — the number of test cases.\n\nThe next 2t lines contain the descriptions of the test cases. \n\nThe first line of each test case description contains an integer n (2 ≤ n ≤ 50) — the length of the array. The second line consists of space-separated elements of the array a_i (-10^9 ≤ a_i ≤ 10^9). Elements of array a may repeat and don't have to be unique.\n\nOutput\n\nPrint t answers to all input test cases. \n\nThe first line of the answer of each test case should contain an integer k (0 ≤ k ≤ n) — the number of actions to sort the array. The next k lines should contain descriptions of the actions formatted as \"l r d\" (without quotes) where l and r (1 ≤ l < r ≤ n) are the boundaries of the segment being shifted, while d (1 ≤ d ≤ r - l) is the offset value. Please remember that only the cyclic shifts to the left are considered so the chosen segment will be shifted by the offset d to the to the left.\n\nNote that you are not required to find the minimum number of cyclic shifts needed for sorting. Any sorting method where the number of shifts does not exceed n will be accepted.\n\nIf the given array a is already sorted, one of the possible answers is k = 0 and an empty sequence of cyclic shifts.\n\nIf there are several possible answers, you may print any of them.\n\nExample\n\nInput\n\n\n4\n2\n2 1\n3\n1 2 1\n4\n2 4 1 3\n5\n2 5 1 4 3\n\n\nOutput\n\n\n1\n1 2 1\n1\n1 3 2\n3\n2 4 1\n2 3 1\n1 3 2\n4\n2 4 2\n1 5 3\n1 2 1\n1 3 1\n\nNote\n\nExplanation of the fourth data set in the example: \n\n  1. The segment a[2 … 4] is selected and is shifted to the left by 2: [2, \\color{blue}{5, 1, 4}, 3] \\longrightarrow [2, \\color{blue}{4, 5, 1}, 3] \n  2. The segment a[1 … 5] is then selected and is shifted to the left by 3: [\\color{blue}{2, 4, 5, 1, 3}] \\longrightarrow [\\color{blue}{1, 3, 2, 4, 5}] \n  3. After that the segment a[1 … 2] is selected and is shifted to the left by 1: [\\color{blue}{1, 3}, 2, 4, 5] \\longrightarrow [\\color{blue}{3, 1}, 2, 4, 5] \n  4. And in the end the segment a[1 … 3] is selected and is shifted to the left by 1: [\\color{blue}{3, 1, 2}, 4, 5] \\longrightarrow [\\color{blue}{1, 2, 3}, 4, 5]"
                     ]
        print('context account:', len(arr_context))
        for i in range(len(arr_context)):
            acc = 0
            results = []
            while acc < 2:
                #context = input("Type input:")
                context = arr_context[i]
                tokens = tokenizer.encode(context)

                start = time.time()

                provided_ctx = len(tokens)
                pad_amount = seq - provided_ctx

                padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
                batched_tokens = np.array([padded_tokens] * total_batch)
                length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

                output = network.generate(batched_tokens, length, 1024, {"top_p": np.ones(total_batch) * 0.9,
                                                                        "temp": np.ones(total_batch) * 0.75})

                for idx, o in enumerate(output[1][0][:, :, 0]):
                    res = 'sample '+str(idx)+': '+repr(tokenizer.decode(o))
                    results.append(res)
                    # print(f"sample {idx}: {repr(tokenizer.decode(o))}")

                print(f"completion done in {time.time() - start:06}s")

            content = '<|endoftext|>'.join(results)
            save('codecontest-bucket','test_result',str(i),content)
