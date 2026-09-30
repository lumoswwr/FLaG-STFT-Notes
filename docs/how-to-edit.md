# Markdown 使用说明

这份项目以后只需要维护 `docs/` 里面的 `.md` 文件，不需要手写 HTML。

## 1. 标题

```markdown
# 一级标题
## 二级标题
### 三级标题
```

在实验页面中，通常一个实验使用二级标题：

```markdown
## S6
```

## 2. 普通文字与粗体

```markdown
普通文字

**这段会加粗**
```

## 3. 列表

```markdown
- 第一项
- 第二项
- 第三项
```

## 4. 表格

```markdown
| Model | AP | F1 |
| --- | ---: | ---: |
| FLaG | 0.7131 | 0.6507 |
| E12 | 0.7613 | 0.6853 |
```

## 5. 图片

把图片放进：

```text
docs/assets/images/
```

然后在 Markdown 中写：

```markdown
![图片说明](../assets/images/example.png)
```

如果当前 Markdown 文件在 `docs/` 根目录，则路径写成：

```markdown
![图片说明](assets/images/example.png)
```

## 6. 数学公式

行内公式：

```markdown
当 $a=b$ 时，反射项消失。
```

独立公式：

```markdown
$$
y[t] = \alpha x[t] + \beta x[(-t) \bmod N]
$$
```

## 7. 新增一个实验

例如以后新增 S6，只需要打开：

```text
docs/sprint/02-experiments.md
```

在文件末尾追加：

```markdown
## S6

**Q:**

这里写问题。

**M:**

这里写方法。

**R:**

这里写结果。

**C:**

这里写结论。
```

保存后，`mkdocs serve` 启动的网页会自动刷新。

## 8. 最常用的三个命令

```bash
mkdocs serve
```

本地实时预览。

```bash
mkdocs build
```

生成最终 HTML 网站到 `site/`。

```bash
Ctrl + C
```

在终端中停止 `mkdocs serve`。
