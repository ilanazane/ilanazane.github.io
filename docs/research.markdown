---
layout: page
title: Resarch
permalink: /research/
order: 2
---

this is a page about my past and current research 

{% assign sorted_pages = site.pages | sort:"order" %}
{% for node in sorted_pages %}
  <li><a href="{{node.url}}">{{node.title}}</a></li>
{% endfor %}
