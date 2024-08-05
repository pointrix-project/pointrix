---
sd_hide_title: true
---

# Pointrix

::::::{div} landing-title
:style: "padding: 0.1rem 0.5rem 0.6rem 0; background-image: linear-gradient(315deg, #bebebe 0%, #919293 74%); clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem)); -webkit-clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem));"

::::{grid}
:reverse:
:gutter: 2 3 3 3
:margin: 4 4 1 2

:::{grid-item}
:columns: 12 5 5 4

```{image} ../images/pointrix_portrait_all_white.png
:class: sd-m-auto sd-animate-grow50-rot20
:width: 150px
```
:::

:::{grid-item}
:columns: 12 8 8 8
:child-align: justify
:class: sd-text-white sd-fs-3

A differentiable point-based rendering library

```{button-ref} get_started/installation
:ref-type: doc
:outline:
:color: white
:class: sd-px-4 sd-fs-5

Get Started
```

:::
::::

::::::

## Overview

[Pointrix](https://github.com/pointrix-project/pointrix) is a **differentiable point-based rendering library** which has following properties:

---

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Highly Extensible
:link: tutorial/tutorial_1
:link-type: doc

Pointrix adopts a modular design, with clear structure and easy extensibility. 

+++
[Learn more »](tutorial/tutorial_1)
:::

:::{grid-item-card} {octicon}`device-camera` Rich Feature
:link: framework/overall
:link-type: doc

Pointrix supports the implementation of various types of tasks.

+++
[Learn more »](framework/overall)
:::

:::{grid-item-card} {octicon}`rocket` Powerful backend
:link: https://github.com/pointrix-project/msplat
:link-type: url

Msplat which offer foundational functionalities for point rendering serves as the backend of Pointrix.

+++
[Learn more »](https://github.com/pointrix-project/msplat)
:::

::::

If you **are the beginner** of Pointrix, You can start from these:

---


::::{grid} 1 1 2 2


:::{grid-item-card}
:padding: 2
:columns: 6
**Get started**
^^^

```{toctree}
:caption: Get started (In 10 minutes)
:maxdepth: 1
get_started/installation
get_started/run_first_model
get_started/render_novel_view_gui
tutorial/learning_configuration
```
:::

:::{grid-item-card}
:padding: 2
:columns: 6
**Apply Pointrix on your research**
^^^

```{toctree}
:maxdepth: 1
:caption: Tutorial
tutorial/tutorial_1
tutorial/tutorial_2
```
:::

::::


If you want to learning more about Pointrix, you can read following:

---

::::{grid} 1 1 2 2


:::{grid-item-card}
:padding: 2
:columns: 6
**Framework introduction of Pointrix**
^^^

```{toctree}
:caption: Framework
:maxdepth: 1
framework/overall
framework/data
framework/model
framework/trainer
framework/hook
```
:::

:::{grid-item-card}
:padding: 2
:columns: 6
**Advanced Usage**
^^^

```{toctree}
:maxdepth: 1
:caption: Advanced
advanced_usage/camera_optimization
advanced_usage/dust3r_init
advanced_usage/extract_mesh
```

```{toctree}
:maxdepth: 1
:caption: Reference
API
```

:::

::::

## Source code and supported methods

|                                                            |                        |
| ---------------------------------------------------------- | ---------------------- |
| [Github](https://github.com/pointrix-project/pointrix)     | Official Pointrix      |
| [Github](https://github.com/pointrix-project/dptr)         | DPTR                   |

## Contributors:
### Pointrix:
<a href="https://github.com/pointrix-project/pointrix/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pointrix-project/pointrix" />
</a>


### DPTR:
<a href="https://github.com/pointrix-project/dptr/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pointrix-project/dptr" />
</a>



```{toctree}
:maxdepth: 2
:hidden: true
switch
```