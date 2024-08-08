---
sd_hide_title: true
---

# Pointrix 中文文档

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

一个轻量可微分点云渲染框架

```{button-ref} get_started_cn/installation
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

[Pointrix](https://github.com/pointrix-project/pointrix) 是一个 **可微分点云渲染框架** 并具有以下几个特点:

---

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` 高度可扩展性
:link: tutorial_cn/tutorial_2
:link-type: doc

Pointrix 采用模块化设计，具有清晰的结构和强大的扩展性。

+++
[了解更多 »](tutorial_cn/tutorial_2)
:::

:::{grid-item-card} {octicon}`device-camera` 丰富的特性
:link: framework_cn/overall
:link-type: doc

Pointrix 支持各种不同类型的工作实现，包括静态动态场景重建，生成以及PBR渲染。

+++
[了解更多 »](framework_cn/overall)
:::

:::{grid-item-card} {octicon}`rocket` 强大的后端
:link: https://github.com/pointrix-project/msplat
:link-type: url

MSplat 作为Pointrix的后端，支持各类点云渲染功能：包括各类特征的渲染，超高阶的sh阶数，以及所有输入的梯度反传。

+++
[了解更多 »](https://github.com/pointrix-project/msplat)
:::

::::
如果你想初步了解Pointrix，你可以从这些开始：

---


::::{grid} 1 1 2 2


:::{grid-item-card}
:padding: 2
:columns: 6
**Get started**
^^^

```{toctree}
:caption: 开始（10分钟内学习Pointrix）
:maxdepth: 1
get_started_cn/installation
get_started_cn/run_first_model
get_started_cn/learning_config
get_started_cn/render_novel_view_gui
```
:::

:::{grid-item-card}
:padding: 2
:columns: 6
**将Pointrix 应用到科研/项目中**
^^^

```{toctree}
:maxdepth: 1
:caption: 教程
tutorial_cn/tutorial_2
```
:::

::::


如果你想进一步了解Poinrtix, 你可以阅读以下文档：

---

::::{grid} 1 1 2 2


:::{grid-item-card}
:padding: 2
:columns: 6
**Pointrix的架构介绍**
^^^

```{toctree}
:caption: 架构
:maxdepth: 1
framework_cn/overall
framework_cn/data
framework_cn/model
framework_cn/trainer
framework_cn/hook
```
:::

:::{grid-item-card}
:padding: 2
:columns: 6
**高阶用法**
^^^

```{toctree}
:maxdepth: 1
:caption: Advanced
advanced_usage_cn/camera_optimization
advanced_usage_cn/dust3r_init
advanced_usage_cn/extract_mesh
```

```{toctree}
:maxdepth: 1
:caption: 参考
API
```
:::

::::

## 贡献者:
### Pointrix:
<a href="https://github.com/pointrix-project/pointrix/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pointrix-project/pointrix" />
</a>


### Msplat:
<a href="https://github.com/pointrix-project/dptr/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pointrix-project/dptr" />
</a>

## Source code

|                                                            |                        |
| ---------------------------------------------------------- | ---------------------- |
| [Github Page](https://github.com/pointrix-project/pointrix)     | Pointrix      |
| [Github Page](https://github.com/pointrix-project/msplat)         | Msplat                   |

```{toctree}
:maxdepth: 1
:hidden: true
switch_cn
```

