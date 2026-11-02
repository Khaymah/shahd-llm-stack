# Data Spec — الـ corpus يُصمَّم قبل أن يُجمع

هذا الملف يشرح `Data/Spec.json`: العقد المكتوب الذي يحدد **ماذا** يتعلّم Shahd و**بأي نسب**، بدل
`--Query="language:typescript stars:>500"`. الخطة الأصلية في `Plan/Brain-Strengthening.md` §2 و§8.
الـ schema في `Foundry/DataSpec.ts`، والجمع في `Foundry/SpecIngest.ts` + `Foundry/GitHubRepoListProvider.ts`،
والـ CLI في `Scripts/IngestSpec.ts`.

## ما هو الـ spec

النطاق (§8): **مساعد تطوير ويب** يُدرَّس على مراحل (curriculum):
**M-A** TS/JS → **M-B** HTML/CSS + design systems → **M-C** Go → **M-D** SQL/DB، ومعها الـ docs التي تشرحها.

| الحقل | المعنى |
|---|---|
| `Slices` | كل slice له `TargetMb` (الحجم المطلوب في الـ Filtered tier) و`Langs` (الـ tags التي يصدرها `LangForPath`، و`markdown`/`text-en` للنصوص). اختياري: `Weight` لنسبة التدريب (الافتراضي = `TargetMb`). |
| `Repos` | القائمة اليدوية. كل entry: `Repo` (`owner/name`)، `Slice`، `Ref` اختياري، `Include`/`Exclude` (globs نسبةً لجذر الريبو)، و`Note` إلزامية عملياً (لماذا هذا الريبو هنا). |
| `Exclusions` | قواعد عامة فوق فلتر الكود المدمج: `PathGlobs`، و`MinFileBytes`/`MaxFileBytes` (200 B – 50 KB). |
| `Training` | `MaxEpochs: 4` (حارس الحفظ)، `ContextTokens: 1024`، `VocabSize: 16384`. |

الريبو الواحد يمكن أن يظهر **مرة لكل slice** (مثلاً `vitejs/vite` في `ts` و`html` و`docs`)؛ الريبو يُنزَّل
مرة واحدة ويُوزَّع على entries. الـ entry بلا `Slice` (حالياً `lodash/lodash`) يأخذ slice كل ملف من لغته.

### الأرقام الحالية (v1)

| slice | TargetMb | entries |
|---|---|---|
| ts | 300 | 28 |
| js | 120 | 16 (+ `lodash/lodash` بلا slice ثابت) |
| html | 40 | 7 |
| css | 40 | 10 |
| go | 150 | 15 |
| sql | 20 | 5 |
| docs | 150 | 18 |

المجموع 100 entry على 77 ريبو مختلف. 16 entry تحتفظ بالـ tests عمداً (مكتوب في `Note`: "Tests KEPT") لتحقيق
نسبة ~10% tests التي تطلبها الخطة.

## كيف تختار ريبو (curation)

- **جودة الكود لا النجوم.** اسأل: هل هو idiomatic؟ هل الـ modules صغيرة وواضحة؟ هل فيه JSDoc/تعليقات تشرح
  "لماذا"؟ هل الـ tests تقرأ كمواصفات؟ ريبو بـ 5K نجمة ومكتوب جيداً أفضل من 100K نجمة بكود مولَّد.
- **أقل وأفضل.** كل entry يجب أن تستحق مكانها؛ اكتب السبب في `Note`. لا تضف ريبو لمجرد زيادة الحجم.
- **ضيّق بـ `Include`.** استهدف الـ source الحقيقي (`src/**`، `packages/*/src/**`، `lib/**`) واترك الـ
  build scripts والـ playgrounds.
- **الـ tests قرار صريح.** الـ spec يُرجع الـ tests (`test/`، `__tests__/`، `*.test.*`، `*_test.go`)؛ لإسقاطها
  اكتب `Exclude: ["**/*.test.*", "**/__tests__/**"]` (أو `["**/*_test.go"]` في Go). احتفظ بها في ريبوهات
  تكون فيها الـ tests نموذجية واذكر ذلك في `Note`.
- **مجلدات docs/examples/migrations/site** يسقطها الفلتر المدمج افتراضياً، وتعود **فقط** لو الـ `Include` الذي
  طابق الملف يذكرها حرفياً (`"docs/**"`، `"examples/**"`، `"**/migrations/**"`).
- مسار غير متأكد منه → `Note: "REVIEW PATHS"`. ترخيص غير متأكد منه → `Note: "REVIEW LICENSE"`.

## قواعد الاستبعاد (بالترتيب الذي تُطبَّق به)

1. `Exclusions.PathGlobs` العامة: fixtures، snapshots، `__mocks__`، i18n/locales، `*.min.*`، `*.d.ts`،
   `*.stories.*`، benchmarks، `vendor`/`third_party`/`node_modules`/`dist`، الملفات المولَّدة، `CHANGELOG`/`LICENSE`.
2. `Exclude` الخاص بالـ entry، ثم `Include` (لو موجود يجب أن يطابق).
3. لغة الملف يجب أن تنتمي لـ slice في الـ spec (ولـ `Slice` الـ entry إن حُدِّد).
4. الحجم الخام بين `MinFileBytes` و`MaxFileBytes` (الملف الأكبر يُسقط، لا يُقطع).
5. الفلتر المدمج `IsSubstantiveCodePath` (dotfiles، config، lockfiles، vendored…) ثم `PrepareRepoFile`:
   فحص header الترخيص لكل ملف + بوابة المحتوى `IsSubstantiveCodeContent`.

ملاحظة: بوابة المحتوى المدمجة تطلب ≥ 300 حرف بعد الـ trim، فالحد الأدنى الفعلي للكود أعلى قليلاً من 200 B.

## سياسة التراخيص

- **Permissive فقط:** MIT، Apache-2.0، BSD-2/3-Clause، ISC، 0BSD، Unlicense، Zlib (`LicenseManifest`).
  ترخيص الريبو يأتي من GitHub API؛ `NOASSERTION` يُحَل بقراءة نص `LICENSE` عبر `SpdxDetector`، ويُقبل فقط
  لو طابق قالباً permissive نظيفاً. ريبو ترخيصه غير permissive **يُتخطّى قبل التنزيل** مع سطر log يطلب إصلاح الـ spec.
- **Header كل ملف هو البوابة الأخيرة:** ملف بـ header من نوع GPL/AGPL/LGPL/MPL/CC-BY-SA/BSL/Commons-Clause
  داخل ريبو MIT يُخزَّن **Rejected** (`file_license_conflict`) بمحتواه الخام كدليل، ولا يدخل التدريب.
- **الـ vendored code مستبعد** (`vendor`، `third_party`، `node_modules`، …) لأنه تحت ترخيصه هو لا ترخيص الريبو.
- **مستبعد صراحةً من الـ spec ولماذا:**
  - `mdn/content` — CC-BY-SA-2.5 (ShareAlike: copyleft على الـ docs).
  - `microsoft/TypeScript-Website` — محتوى الـ handbook CC-BY-4.0، وهو ليس في قائمة الـ permissive عندنا.
  - `tailwindlabs/tailwindcss.com` — موقع الـ docs بلا ترخيص مفتوح.
  - `golang/website` و`doc/` في `golang/go` — محتوى go.dev تحت CC-BY-3.0؛ نأخذ كود الـ stdlib (BSD-3) فقط.
  - `oven-sh/bun` — مستودع مختلط (أجزاء WebKit/JSC تحت LGPL)؛ لا يستحق المخاطرة.
  - `mdn/learning-area`، `mdn/dom-examples` — CC0، خارج القائمة الحالية.
  - `mrdoob/three.js` — MIT لكن الـ tarball (examples/models) يتجاوز سقف `MaxTarballBytes`.
  - `web-platform-tests/wpt` — tests فقط، ليس كوداً تعليمياً.

## التشغيل

```bash
bun run foundry:ingest-spec --Status                    # جدول: slice -> target / collected / % / files / repos done
bun run foundry:ingest-spec --DryRun                    # ماذا سيُجلب (لا شبكة، لا كتابة)
bun run foundry:ingest-spec --DryRun --Slice=go         # خطة slice واحد
bun run foundry:ingest-spec                             # اجمع كل ما لم يُجمع بعد
bun run foundry:ingest-spec --Slice=ts                  # slice واحد
bun run foundry:ingest-spec --Repo=honojs/hono --Force  # ريبو واحد، حتى لو جُمع سابقاً
bun run foundry:ingest-spec --Spec=Data/Spec.json --Store=postgres
```

- `--Spec` (أو `FOUNDRY_SPEC`) — الافتراضي `Data/Spec.json`.
- `--Store=memory|postgres` — مثل باقي scripts الـ foundry؛ مع Postgres الصفوف تذهب لجدول `documents_code`.
- `GITHUB_TOKEN` موصى به بشدة: كل ريبو = طلبان (metadata + tarball)، والـ API بدون token ~60 طلب/ساعة.
- `--Slice`/`--Repo` بقيمة غير موجودة في الـ spec = خطأ صريح (لا تشغيل صامت بلا نتيجة).
- entry تُعتبر "done" لو الـ store فيه صفوف لهذا الريبو بلغات الـ slice (أي tier). تُتخطّى إلا مع `--Force`
  (والـ dedup بالـ content hash يمنع العدّ المزدوج). entry لم تُنتج أي ملف تُعاد في كل تشغيل — أصلح `Include`.
- البوابة `Config.Data.WebEnabled` تُطبَّق على الجمع الفعلي (لا على `--Status`/`--DryRun`).

## كيف يعمل `TargetMb`

- `TargetMb` = بايتات المحتوى المخزَّن (بعد نزع header الترخيص والتنظيف) في الـ **Filtered** tier فقط،
  بوحدة MiB (2^20 بايت). الـ Rejected والـ Raw لا يُحسبان.
- التقدّم يُقاس من الـ store نفسه (`RepoSummaries` لكل لغة في الـ slice)، لا من عدّاد في الذاكرة، فهو صحيح عبر
  التشغيلات وبعد أي إعادة تصنيف.
- قبل تنزيل أي ريبو يُفحص: هل الـ slice ممتلئ؟ لو نعم يُتخطّى بلا تنزيل. بعد تخزين كل ريبو يُعاد القياس، والـ
  slice الممتلئ يتوقف أيضاً عن قبول ملفات من الريبوهات المختلطة. الريبو الجاري يُكمَل، فالـ slice قد يتجاوز
  هدفه بحجم ريبو واحد على الأكثر.
- الترتيب في `Repos` هو الأولوية: ضع الأهم أولاً داخل كل slice.
