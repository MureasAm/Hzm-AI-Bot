# -*- coding: utf-8 -*-
"""显式 fail-open 触发：只认言语行为，不认话题。"""
from src.plugins.chatbot import need_triggers as nt


def _kinds(msg):
    return [r.get("kind") for r in nt.requests_for(msg)]


def test_recall_identity():
    assert "fact" in _kinds("你还记得我叫什么吗")
    assert "fact" in _kinds("我是谁")


def test_address_negotiation():
    ks = _kinds("我想亲昵点，你想怎么称呼我")
    assert "fact" in ks and "interaction_preference" in ks


def test_weather_owner():
    reqs = nt.requests_for("你那边冷吗")
    assert any(r.get("kind") == "weather" and r.get("subject") == "her" for r in reqs)
    reqs = nt.requests_for("我这边下雨了吗")
    assert any(r.get("kind") == "weather" and r.get("subject") == "user" for r in reqs)


def test_today_and_perception():
    assert "daily" in _kinds("我早上吃了鸡蛋")
    assert "daily" in _kinds("我今天干嘛了")
    assert "perception" in _kinds("现在几点了")


def test_no_false_positive():
    for msg in ("我好想你", "晚安", "我每天跑步", "我喜欢的颜色是蓝色", "冷静一下"):
        assert nt.requests_for(msg) == [], msg


def test_place_from_text():
    assert nt.place_from_text("坡县几度啊") == "新加坡"
    assert nt.place_from_text("用户在问新加坡现在的气温是多少度") == "新加坡"
    assert nt.place_from_text("广东今天天气怎么样") == "广东"
    assert nt.place_from_text("今天天气不错") == ""
    assert nt.place_from_text("你那边天气好吗") == ""