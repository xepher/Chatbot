# encoding=utf-8
import pytest
from pre_process import parse_ptt_lines

def test_parse_ptt_lines_valid():
    lines = [
        "为什么庆祝228会被骂可是庆端午不会？\t因为屈原不是台湾人，是楚國人。\n",
        " 有没有战神阿瑞斯的八卦?\t爵士就是阿瑞斯 男主角最后死了 ",
    ]
    result = parse_ptt_lines(lines)
    assert len(result) == 4
    assert result[0] == "为什么庆祝228会被骂可是庆端午不会？"
    assert result[1] == "因为屈原不是台湾人，是楚國人。"
    assert result[2] == "有没有战神阿瑞斯的八卦?"
    assert result[3] == "爵士就是阿瑞斯 男主角最后死了"

def test_parse_ptt_lines_invalid_cases():
    lines = [
        # Missing answer
        "只有一个问题\n",
        # Empty line
        "   \n",
        # Extra columns
        "问题\t答案\t多余内容\n",
        # Empty fields
        "问题\t\n",
        "\t答案\n",
    ]
    result = parse_ptt_lines(lines)
    assert len(result) == 0
