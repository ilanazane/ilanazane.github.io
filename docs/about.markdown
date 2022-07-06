---
layout: page
title: About
permalink: /about/
order: 1
---

## Eduation 

Ilana Zane has her B.S. in computer science with a minor in french literature from Rutgers, New Brunswick. She is currently at Stevens Institute of Technology pursuing her M.S. in Applied AI and Mechanical Engineering-- which means I take a lot of machine learning classes, some robotics classes and when the two overlap I am super happy :) 

## Personal Intersts 

Ilana's hobbies include 3D Printing ( she has an Ender 3V2 which has been amazing), bullet journaling, gardening, and playing tennis. 

{% assign sorted_pages = site.pages | sort:"order" %}
{% for node in sorted_pages %}
  <li><a href="{{node.url}}">{{node.title}}</a></li>
{% endfor %}

